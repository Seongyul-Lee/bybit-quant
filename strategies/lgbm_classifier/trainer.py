"""Walk-Forward 학습 + Optuna 튜닝 모듈.

확장 윈도우 방식으로 시계열 데이터를 분할하고,
LightGBM 모델을 학습/검증한다.
"""

import json
import os
import shutil
import tempfile
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from src.utils.logger import setup_logger

logger = setup_logger("lgbm_trainer")

# 라벨 매핑: 원본(-1,0,1) → 학습용(0,1,2) / 역매핑(0,1,2) → 원본(-1,0,1)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}


class WalkForwardTrainer:
    """Walk-Forward 방식으로 LightGBM 모델을 학습.

    확장 윈도우: 최소 train_months 학습 → val_months 검증.
    Embargo: 학습/검증 사이 embargo_bars 제거.
    Optuna: 첫 fold에서만 튜닝 → 이후 fold에 동일 파라미터 적용.

    Attributes:
        min_train_months: 최소 학습 기간 (월).
        val_months: 검증 기간 (월).
        embargo_bars: 학습/검증 사이 제거할 봉 수.
        n_optuna_trials: Optuna 시행 횟수.
    """

    # LightGBM 고정 파라미터
    FIXED_PARAMS: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 1000,
        "bagging_freq": 5,
        "class_weight": "balanced",
        "verbose": -1,
    }

    def __init__(
        self,
        min_train_months: int = 6,
        val_months: int = 1,
        embargo_bars: int = 24,
        n_optuna_trials: int = 50,
    ) -> None:
        """WalkForwardTrainer 초기화.

        Args:
            min_train_months: 최소 학습 데이터 기간 (월).
            val_months: 각 fold의 검증 기간 (월).
            embargo_bars: 학습 끝~검증 시작 사이 제거할 봉 수.
            n_optuna_trials: Optuna 하이퍼파라미터 탐색 시행 수.
        """
        self.min_train_months = min_train_months
        self.val_months = val_months
        self.embargo_bars = embargo_bars
        self.n_optuna_trials = n_optuna_trials

    def generate_folds(self, df: pd.DataFrame) -> list[dict]:
        """Walk-Forward fold를 생성.

        확장 윈도우: 학습 시작은 고정, 학습 끝은 매 fold마다 val_months씩 확장.

        Args:
            df: timestamp 컬럼이 있는 데이터프레임.

        Returns:
            fold 리스트. 각 fold는 dict:
            {"train_start", "train_end", "val_start", "val_end",
             "train_idx", "val_idx"} 인덱스 범위.
        """
        timestamps = pd.to_datetime(df["timestamp"])
        start_date = timestamps.iloc[0]
        end_date = timestamps.iloc[-1]

        # 최소 학습 기간 이후 첫 검증 시작점
        first_val_start = start_date + pd.DateOffset(months=self.min_train_months)

        folds = []
        val_start = first_val_start

        while True:
            val_end = val_start + pd.DateOffset(months=self.val_months)
            if val_end > end_date:
                break

            train_mask = timestamps < val_start
            # embargo 적용: 학습 끝에서 embargo_bars 제거
            train_idx = df.index[train_mask].tolist()
            if len(train_idx) > self.embargo_bars:
                train_idx = train_idx[:-self.embargo_bars]

            val_mask = (timestamps >= val_start) & (timestamps < val_end)
            val_idx = df.index[val_mask].tolist()

            if len(train_idx) < 100 or len(val_idx) < 10:
                val_start = val_end
                continue

            folds.append({
                "train_start": str(timestamps.iloc[train_idx[0]]),
                "train_end": str(timestamps.iloc[train_idx[-1]]),
                "val_start": str(val_start),
                "val_end": str(val_end),
                "train_idx": train_idx,
                "val_idx": val_idx,
            })

            val_start = val_end

        logger.info(f"Walk-Forward fold 생성: {len(folds)}개")
        return folds

    def optimize_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Optuna로 하이퍼파라미터 최적화.

        Args:
            X_train: 학습 피처.
            y_train: 학습 라벨 (0, 1, 2).
            X_val: 검증 피처.
            y_val: 검증 라벨.

        Returns:
            최적 하이퍼파라미터 딕셔너리.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                **self.FIXED_PARAMS,
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average="macro")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optuna_trials)

        best_params = {**self.FIXED_PARAMS, **study.best_params}
        logger.info(f"Optuna 최적 F1(macro): {study.best_value:.4f}")
        logger.info(f"최적 파라미터: {study.best_params}")

        return best_params

    def train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple[lgb.LGBMClassifier, dict]:
        """단일 fold 학습.

        Args:
            X_train: 학습 피처.
            y_train: 학습 라벨.
            X_val: 검증 피처.
            y_val: 검증 라벨.
            params: LightGBM 파라미터.

        Returns:
            (학습된 모델, fold 성과 딕셔너리).
        """
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)

        train_f1 = f1_score(y_train, y_pred_train, average="macro")
        val_f1 = f1_score(y_val, y_pred_val, average="macro")
        val_logloss = log_loss(y_val, y_proba_val)

        fold_metrics = {
            "train_f1_macro": float(train_f1),
            "val_f1_macro": float(val_f1),
            "val_logloss": float(val_logloss),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": model.best_iteration_ if hasattr(model, "best_iteration_") else -1,
        }

        logger.info(
            f"  Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
            f"Val LogLoss: {val_logloss:.4f}"
        )

        return model, fold_metrics

    def run(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        label_col: str = "label",
    ) -> dict:
        """전체 Walk-Forward 학습 파이프라인 실행.

        Args:
            df: 피처 + 라벨이 포함된 데이터프레임.
            feature_names: 사용할 피처 컬럼 이름 목록.
            label_col: 라벨 컬럼 이름.

        Returns:
            결과 딕셔너리:
            {"model", "params", "feature_names", "folds_metrics", "feature_importance"}.
        """
        folds = self.generate_folds(df)
        if not folds:
            raise ValueError("fold를 생성할 수 없습니다. 데이터가 충분한지 확인하세요.")

        # 라벨 매핑 (-1→0, 0→1, 1→2)
        df = df.copy()
        df[label_col] = df[label_col].map(LABEL_MAP)

        # 첫 fold에서 Optuna 튜닝
        first_fold = folds[0]
        X_train_0 = df.loc[first_fold["train_idx"], feature_names]
        y_train_0 = df.loc[first_fold["train_idx"], label_col]
        X_val_0 = df.loc[first_fold["val_idx"], feature_names]
        y_val_0 = df.loc[first_fold["val_idx"], label_col]

        if self.n_optuna_trials > 0:
            logger.info(f"Fold 0: Optuna 튜닝 ({self.n_optuna_trials} trials)...")
            best_params = self.optimize_hyperparams(X_train_0, y_train_0, X_val_0, y_val_0)
        else:
            best_params = {
                **self.FIXED_PARAMS,
                "num_leaves": 31,
                "min_child_samples": 50,
                "learning_rate": 0.05,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
            }

        # 모든 fold에서 학습
        folds_metrics = []
        last_model = None

        for i, fold in enumerate(folds):
            logger.info(
                f"Fold {i}/{len(folds) - 1}: "
                f"train={fold['train_start']}~{fold['train_end']} | "
                f"val={fold['val_start']}~{fold['val_end']}"
            )

            X_train = df.loc[fold["train_idx"], feature_names]
            y_train = df.loc[fold["train_idx"], label_col]
            X_val = df.loc[fold["val_idx"], feature_names]
            y_val = df.loc[fold["val_idx"], label_col]

            model, metrics = self.train_fold(X_train, y_train, X_val, y_val, best_params)
            metrics["fold"] = i
            metrics["train_period"] = f"{fold['train_start']} ~ {fold['train_end']}"
            metrics["val_period"] = f"{fold['val_start']} ~ {fold['val_end']}"
            folds_metrics.append(metrics)
            last_model = model

        # 피처 중요도 (마지막 fold 모델 기준)
        importance = dict(zip(
            feature_names,
            [int(x) for x in last_model.feature_importances_],
        ))

        return {
            "model": last_model,
            "params": best_params,
            "feature_names": feature_names,
            "folds_metrics": folds_metrics,
            "feature_importance": importance,
        }

    def save_model(
        self,
        model: lgb.LGBMClassifier,
        params: dict,
        feature_names: list[str],
        folds_metrics: list[dict],
        feature_importance: dict,
        save_dir: str,
    ) -> str:
        """학습 결과를 atomic write로 저장.

        Args:
            model: 학습된 LGBMClassifier.
            params: 사용된 하이퍼파라미터.
            feature_names: 피처 이름 목록.
            folds_metrics: fold별 성과.
            feature_importance: 피처 중요도.
            save_dir: 저장 디렉토리 경로.

        Returns:
            모델 파일 경로.
        """
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "latest.txt")
        feature_names_path = os.path.join(save_dir, "feature_names.json")
        params_path = os.path.join(save_dir, "best_params.json")
        meta_path = os.path.join(save_dir, "training_meta.json")

        # atomic write: tempfile → shutil.move
        self._atomic_write_text(model_path, model.booster_.model_to_string())

        self._atomic_write_json(feature_names_path, feature_names)

        # params에서 직렬화 불가능한 값 정리
        serializable_params = {
            k: v for k, v in params.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        self._atomic_write_json(params_path, serializable_params)

        meta = {
            "n_folds": len(folds_metrics),
            "folds_metrics": folds_metrics,
            "feature_importance": feature_importance,
        }
        self._atomic_write_json(meta_path, meta)

        logger.info(f"모델 저장 완료: {save_dir}")
        return model_path

    @staticmethod
    def _atomic_write_text(path: str, content: str) -> None:
        """텍스트 파일 atomic write."""
        dir_path = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            shutil.move(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    @staticmethod
    def _atomic_write_json(path: str, data: Any) -> None:
        """JSON 파일 atomic write."""
        dir_path = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise
