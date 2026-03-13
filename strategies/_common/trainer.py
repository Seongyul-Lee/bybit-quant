"""Walk-Forward 학습 + Optuna 튜닝 모듈.

확장 윈도우 방식으로 시계열 데이터를 분할하고,
LightGBM 모델을 학습/검증한다.
분류(classifier)와 회귀(regressor) 모드를 지원한다.
"""

import json
import os
import shutil
import tempfile
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, log_loss

from src.utils.logger import setup_logger

logger = setup_logger("lgbm_trainer")

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

    # LightGBM 분류 고정 파라미터
    FIXED_PARAMS: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 2000,
        "bagging_freq": 1,
        "max_depth": -1,       # 제한 없음 (num_leaves로만 제어)
        "is_unbalance": False,  # 라벨 52/48 거의 균형 → 불필요
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
    }

    # LightGBM 회귀 고정 파라미터
    REGRESSOR_FIXED_PARAMS: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "huber",           # 이상치에 강건
        "metric": "mae",
        "n_estimators": 2000,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
        "huber_delta": 1.0,
    }

    def __init__(
        self,
        mode: str = "classifier",
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
            mode: 모델 유형 — "classifier" 또는 "regressor".
            min_train_months: 최소 학습 데이터 기간 (월).
            val_months: 각 fold의 검증 기간 (월).
            embargo_bars: 학습 끝~검증 시작 사이 제거할 봉 수.
            n_optuna_trials: Optuna 하이퍼파라미터 탐색 시행 수.
            max_overfit_gap: 모델 후보 제외 기준 과적합 갭.
            use_sliding_window: True면 슬라이딩 윈도우, False면 확장 윈도우.
            sliding_window_months: 슬라이딩 윈도우 시 학습 데이터 최대 기간 (월).
        """
        self.mode = mode
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
            y_train: 학습 라벨.
            X_val: 검증 피처.
            y_val: 검증 라벨.

        Returns:
            최적 하이퍼파라미터 딕셔너리.
        """
        if self.mode == "regressor":
            return self._optimize_regressor(X_train, y_train, X_val, y_val)
        return self._optimize_classifier(X_train, y_train, X_val, y_val)

    def _optimize_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Optuna 분류 모드 하이퍼파라미터 탐색."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                **self.FIXED_PARAMS,
                "num_leaves": trial.suggest_int("num_leaves", 15, 31),
                "min_child_samples": trial.suggest_int("min_child_samples", 80, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.7),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.8),
                "max_depth": trial.suggest_int("max_depth", 4, 6),
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            train_f1 = f1_score(y_train, y_pred_train, average="binary", pos_label=1)
            val_f1 = f1_score(y_val, y_pred_val, average="binary", pos_label=1)
            gap = train_f1 - val_f1
            # F1에서 과적합 갭 페널티 차감 → 일반화 성능 우선
            return val_f1 - 0.5 * max(gap - 0.1, 0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optuna_trials)

        best_params = {**self.FIXED_PARAMS, **study.best_params}
        logger.info(f"Optuna 최적 F1(binary): {study.best_value:.4f}")
        logger.info(f"최적 파라미터: {study.best_params}")

        return best_params

    def _optimize_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Optuna 회귀 모드 하이퍼파라미터 탐색.

        목적함수: val_ic - 0.5 * max(mae_gap - 0.002, 0)
          — IC 최대화 + 과적합 페널티.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                **self.REGRESSOR_FIXED_PARAMS,
                "num_leaves": trial.suggest_int("num_leaves", 8, 31),
                "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.8),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
                "huber_delta": trial.suggest_float("huber_delta", 0.5, 2.0),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)

            val_ic, _ = spearmanr(pred_val, y_val)
            if np.isnan(val_ic):
                return -1.0

            train_mae = np.mean(np.abs(pred_train - y_train))
            val_mae = np.mean(np.abs(pred_val - y_val))
            mae_gap = val_mae - train_mae

            return val_ic - 0.5 * max(mae_gap - 0.002, 0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optuna_trials)

        best_params = {**self.REGRESSOR_FIXED_PARAMS, **study.best_params}
        logger.info(f"Optuna 최적 IC: {study.best_value:.4f}")
        logger.info(f"최적 파라미터: {study.best_params}")

        return best_params

    def train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple:
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
        if self.mode == "regressor":
            return self._train_fold_regressor(X_train, y_train, X_val, y_val, params)
        return self._train_fold_classifier(X_train, y_train, X_val, y_val, params)

    def _train_fold_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple[lgb.LGBMClassifier, dict]:
        """분류 fold 학습."""
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)

        train_f1 = f1_score(y_train, y_pred_train, average="binary", pos_label=1)
        val_f1 = f1_score(y_val, y_pred_val, average="binary", pos_label=1)
        val_logloss = log_loss(y_val, y_proba_val, labels=[0, 1])

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

    def _train_fold_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict,
    ) -> tuple[lgb.LGBMRegressor, dict]:
        """회귀 fold 학습.

        평가 메트릭: MAE, IC (Spearman 상관), Directional Accuracy.
        """
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        # MAE
        train_mae = np.mean(np.abs(pred_train - y_train))
        val_mae = np.mean(np.abs(pred_val - y_val))

        # IC (Spearman 상관)
        train_ic, _ = spearmanr(pred_train, y_train)
        val_ic, _ = spearmanr(pred_val, y_val)

        # Directional Accuracy (예측 부호 vs 실제 부호 일치율)
        train_dir_acc = np.mean(np.sign(pred_train) == np.sign(y_train.values))
        val_dir_acc = np.mean(np.sign(pred_val) == np.sign(y_val.values))

        fold_metrics = {
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_ic": float(train_ic) if not np.isnan(train_ic) else 0.0,
            "val_ic": float(val_ic) if not np.isnan(val_ic) else 0.0,
            "train_dir_acc": float(train_dir_acc),
            "val_dir_acc": float(val_dir_acc),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": model.best_iteration_ if hasattr(model, "best_iteration_") else -1,
            # 하위 호환: run()에서 참조하는 키 맞추기
            "train_f1_macro": float(train_ic) if not np.isnan(train_ic) else 0.0,
            "val_f1_macro": float(val_ic) if not np.isnan(val_ic) else 0.0,
            "val_logloss": float(val_mae),
        }

        logger.info(
            f"  Train MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f} | "
            f"Train IC: {train_ic:.4f} | Val IC: {val_ic:.4f} | "
            f"Val DirAcc: {val_dir_acc:.4f}"
        )

        return model, fold_metrics

    def run(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        label_col: str = "label",
        override_params: dict | None = None,
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

        df = df.copy()

        # 첫 fold에서 Optuna 튜닝
        first_fold = folds[0]
        X_train_0 = df.loc[first_fold["train_idx"], feature_names]
        y_train_0 = df.loc[first_fold["train_idx"], label_col]
        X_val_0 = df.loc[first_fold["val_idx"], feature_names]
        y_val_0 = df.loc[first_fold["val_idx"], label_col]

        base_params = self.REGRESSOR_FIXED_PARAMS if self.mode == "regressor" else self.FIXED_PARAMS

        if override_params:
            best_params = {**base_params, **override_params}
            logger.info(f"외부 파라미터 사용: {len(override_params)}개 키")
        elif self.n_optuna_trials > 0:
            logger.info(f"Fold 0: Optuna 튜닝 ({self.n_optuna_trials} trials)...")
            best_params = self.optimize_hyperparams(X_train_0, y_train_0, X_val_0, y_val_0)
        elif self.mode == "regressor":
            best_params = {
                **self.REGRESSOR_FIXED_PARAMS,
                "num_leaves": 15,
                "min_child_samples": 150,
                "learning_rate": 0.02,
                "reg_alpha": 3.0,
                "reg_lambda": 5.0,
                "feature_fraction": 0.5,
                "bagging_fraction": 0.7,
                "max_depth": 4,
                "n_estimators": 200,
            }
        else:
            best_params = {
                **self.FIXED_PARAMS,
                "num_leaves": 15,
                "min_child_samples": 150,
                "learning_rate": 0.02,
                "reg_alpha": 3.0,
                "reg_lambda": 5.0,
                "feature_fraction": 0.5,
                "bagging_fraction": 0.7,
                "max_depth": 4,
                "n_estimators": 200,
            }

        # 모든 fold에서 학습
        folds_metrics = []
        fold_models = []

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

        # 모델 선택: gap 최소 우선 (일반화 성능 최우선)
        # 1. 학습 실패 fold 제외 (train_f1 < 0.01)
        # 2. gap ≤ max_overfit_gap인 fold만 후보
        # 3. 후보 중 Val F1 ≥ 0.40 (최소 품질 기준)
        # 4. 조건 충족 fold 중 gap이 가장 낮은 fold 선택
        # 5. gap 동점이면 최신 fold 우선
        selected_model = None
        selected_fold_idx = -1

        # 회귀 모드에서는 IC 기준 (0.01 이상이면 유의미)
        MIN_VAL_F1 = 0.01 if self.mode == "regressor" else 0.40
        MIN_TRAIN_F1 = 0.001 if self.mode == "regressor" else 0.01

        # 유효 fold 필터링: 학습 성공 + gap ≤ threshold + Val F1 ≥ 최소 기준
        eligible = [
            (i, folds_metrics[i]["val_f1_macro"], folds_metrics[i]["overfit_gap"])
            for i in range(len(folds))
            if folds_metrics[i]["train_f1_macro"] >= MIN_TRAIN_F1
            and folds_metrics[i]["overfit_gap"] <= self.max_overfit_gap
            and folds_metrics[i]["val_f1_macro"] >= MIN_VAL_F1
        ]

        metric_name = "Val IC" if self.mode == "regressor" else "Val F1"

        if eligible:
            # 최신 fold 우선 (더 많은 학습 데이터 + 최근 패턴),
            # 동점이면 Val 메트릭 높은 쪽 우선
            best = max(eligible, key=lambda x: (x[0], x[1]))
            selected_fold_idx = best[0]
            selected_model = fold_models[selected_fold_idx]
            logger.info(
                f"모델 선택: Fold {selected_fold_idx} "
                f"({metric_name}: {best[1]:.4f}, gap: {best[2]:.4f}) "
                f"— {len(eligible)}개 적합 fold 중 최신 fold"
            )
        else:
            # Fallback: 학습 성공 fold 중 gap 최소
            valid = [
                (i, folds_metrics[i]["val_f1_macro"], folds_metrics[i]["overfit_gap"])
                for i in range(len(folds))
                if folds_metrics[i]["train_f1_macro"] >= MIN_TRAIN_F1
            ]
            if valid:
                best = min(valid, key=lambda x: (x[2], -x[0]))
            else:
                best = (len(folds) - 1, folds_metrics[-1]["val_f1_macro"], folds_metrics[-1]["overfit_gap"])
            selected_fold_idx = best[0]
            selected_model = fold_models[selected_fold_idx]
            logger.warning(
                f"적합 fold 없음 — Fold {selected_fold_idx} 사용 "
                f"(gap: {best[2]:.4f}, {metric_name}: {best[1]:.4f})"
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
            "fold_models": fold_models,
        }

    def save_model(
        self,
        model,
        params: dict,
        feature_names: list[str],
        folds_metrics: list[dict],
        feature_importance: dict,
        save_dir: str,
        best_fold_idx: int = -1,
        best_val_f1: float = 0.0,
        fold_models: list | None = None,
    ) -> str:
        """학습 결과를 atomic write로 저장.

        Args:
            model: 학습된 LGBMClassifier 또는 LGBMRegressor.
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
            "mode": self.mode,
            "n_folds": len(folds_metrics),
            "best_fold_idx": best_fold_idx,
            "best_val_f1": best_val_f1,
            "folds_metrics": folds_metrics,
            "feature_importance": feature_importance,
        }
        self._atomic_write_json(meta_path, meta)

        # 모든 fold 모델 저장
        if fold_models:
            for i, fm in enumerate(fold_models):
                fold_path = os.path.join(save_dir, f"fold_{i:02d}.txt")
                self._atomic_write_text(fold_path, fm.booster_.model_to_string())
            logger.info(f"전체 fold 모델 {len(fold_models)}개 저장 완료")

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
