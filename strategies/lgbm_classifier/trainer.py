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
        "bagging_freq": 1,
        "max_depth": 6,
        "class_weight": "balanced",
        "verbose": -1,
    }

    def __init__(
        self,
        min_train_months: int = 6,
        val_months: int = 1,
        embargo_bars: int = 24,
        n_optuna_trials: int = 50,
        max_overfit_gap: float = 0.3,
        use_sliding_window: bool = False,
        sliding_window_months: int = 12,
    ) -> None:
        """WalkForwardTrainer 초기화.

        Args:
            min_train_months: 최소 학습 데이터 기간 (월).
            val_months: 각 fold의 검증 기간 (월).
            embargo_bars: 학습 끝~검증 시작 사이 제거할 봉 수.
            n_optuna_trials: Optuna 하이퍼파라미터 탐색 시행 수.
            max_overfit_gap: 모델 후보 제외 기준 과적합 갭 (train_f1 - val_f1).
            use_sliding_window: True면 슬라이딩 윈도우, False면 확장 윈도우.
            sliding_window_months: 슬라이딩 윈도우 시 학습 데이터 최대 기간 (월).
        """
        self.min_train_months = min_train_months
        self.val_months = val_months
        self.embargo_bars = embargo_bars
        self.n_optuna_trials = n_optuna_trials
        self.max_overfit_gap = max_overfit_gap
        self.use_sliding_window = use_sliding_window
        self.sliding_window_months = sliding_window_months

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

            # 슬라이딩 윈도우: 학습 시작을 제한하여 최근 데이터만 사용
            if self.use_sliding_window:
                train_start_limit = val_start - pd.DateOffset(
                    months=self.sliding_window_months
                )
                train_mask = (timestamps >= train_start_limit) & (timestamps < val_start)
            else:
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

        window_type = "슬라이딩" if self.use_sliding_window else "확장"
        logger.info(f"Walk-Forward fold 생성: {len(folds)}개 ({window_type} 윈도우)")
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
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.8),
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
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
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)

        train_f1 = f1_score(y_train, y_pred_train, average="macro")
        val_f1 = f1_score(y_val, y_pred_val, average="macro")
        val_logloss = log_loss(y_val, y_proba_val, labels=[0, 1, 2])

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
                "num_leaves": 15,              # H-1: 62→15 (과적합 억제)
                "min_child_samples": 100,      # H-3: 44→100 (leaf 분할 제약)
                "learning_rate": 0.05,
                "reg_alpha": 1.5,              # L1 정규화 유지
                "reg_lambda": 2.0,             # H-2: 0.012→2.0 (L2 정규화 활성화)
                "feature_fraction": 0.6,       # H-4: 0.33→0.6 (피처 다양성)
                "bagging_fraction": 0.7,       # 0.52→0.7
            }

        # 모든 fold에서 학습
        folds_metrics = []
        fold_models: list[lgb.LGBMClassifier] = []

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

            # 과적합 갭 계산
            overfit_gap = metrics["train_f1_macro"] - metrics["val_f1_macro"]
            metrics["overfit_gap"] = float(overfit_gap)
            if overfit_gap > self.max_overfit_gap:
                logger.warning(
                    f"  Fold {i}: 과적합 심각 (gap={overfit_gap:.4f} > {self.max_overfit_gap}) "
                    f"— 모델 후보에서 제외"
                )

            folds_metrics.append(metrics)
            fold_models.append(model)

        # 모델 선택: 최신 fold부터 역순으로, gap ≤ threshold인 첫 번째 fold 선택
        # 모두 threshold 초과 시 최신 fold 사용 (경고 출력)
        selected_model = None
        selected_fold_idx = -1

        for i in range(len(folds) - 1, -1, -1):
            gap = folds_metrics[i]["overfit_gap"]
            if gap <= self.max_overfit_gap:
                selected_model = fold_models[i]
                selected_fold_idx = i
                logger.info(
                    f"모델 선택: Fold {i} (Val F1: {folds_metrics[i]['val_f1_macro']:.4f}, "
                    f"gap: {gap:.4f}) — 최신 적합 fold"
                )
                break

        if selected_model is None:
            # 모든 fold가 과적합 — 최신 fold 사용 (가장 많은 데이터로 학습됨)
            selected_model = fold_models[-1]
            selected_fold_idx = len(folds) - 1
            logger.warning(
                f"모든 fold가 과적합 (gap > {self.max_overfit_gap}). "
                f"최신 Fold {selected_fold_idx} 사용 "
                f"(Val F1: {folds_metrics[selected_fold_idx]['val_f1_macro']:.4f})"
            )

        selected_val_f1 = folds_metrics[selected_fold_idx]["val_f1_macro"]

        # 피처 중요도 (선택된 fold 모델 기준)
        importance = dict(zip(
            feature_names,
            [int(x) for x in selected_model.feature_importances_],
        ))

        return {
            "model": selected_model,
            "params": best_params,
            "feature_names": feature_names,
            "folds_metrics": folds_metrics,
            "feature_importance": importance,
            "best_fold_idx": selected_fold_idx,
            "best_val_f1": selected_val_f1,
        }

    def save_model(
        self,
        model: lgb.LGBMClassifier,
        params: dict,
        feature_names: list[str],
        folds_metrics: list[dict],
        feature_importance: dict,
        save_dir: str,
        best_fold_idx: int = -1,
        best_val_f1: float = 0.0,
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
            "best_fold_idx": best_fold_idx,
            "best_val_f1": best_val_f1,
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
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
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
