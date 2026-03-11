---
name: verify-ml
description: ML 전략(LightGBM) 전용 규칙 준수 여부를 검증합니다. ML 전략 코드 수정, 학습 파이프라인 변경, 모델 파일 구조 변경 후 사용. strategies/_common/, strategies/{전략명}/ 내 파일, train_lgbm.py, 또는 ML 관련 config 수정 시 반드시 실행하세요.
---

# ML 전략 규칙 검증

## 검증 항목 요약

| # | 검사 항목 | 핵심 목적 |
|---|-----------|-----------|
| 1 | 이진 분류 고정 | FIXED_PARAMS["objective"] == "binary", is_unbalance == False |
| 2 | 모델 파일 구조 | 각 전략의 models/ 디렉토리에 필수 4개 파일 + fold_XX.txt 존재 |
| 3 | 피처 일관성 | 학습/추론이 동일한 strategies._common.features.FeatureEngine 사용 |
| 4 | Walk-Forward 규칙 | embargo ≥ 24, min_train ≥ 6, Optuna 첫 fold 전용 |
| 5 | Live 재학습 금지 | main.py에 학습 코드 부재 |
| 6 | Atomic Write (모델) | trainer.py save_model의 tempfile → shutil.move 패턴 |
| 7 | config.yaml ML 파라미터 | confidence_threshold 범위, model_path/feature_names_path 경로 정합성 |
| 8 | generate_signal ↔ vectorized 일관성 | 두 메서드의 로직 동치 |
| 9 | 과적합 체크 통합 | train_lgbm.py에서 ModelEvaluator.check_overfitting 호출 |
| 10 | 피처 선별 파이프라인 일관성 | feature_names.json 기반 학습/추론 일관성 |
| 11 | Best Fold 선택 로직 | eligible fold 중 최신 fold 우선 (gap ≤ max_overfit_gap, Val F1 ≥ 0.40) |
| 12 | labeler_type 동적 선택 | config.yaml의 labeler_type에 따라 올바른 labeler 사용 |

## When to Run

- `strategies/_common/` 내 Python 파일을 수정한 후 (features.py, labeler.py, trainer.py, evaluator.py)
- `strategies/{전략명}/strategy.py`를 수정한 후
- `train_lgbm.py`를 수정한 후
- ML 모델 저장/로드 로직을 변경한 후
- `strategies/{전략명}/config.yaml`을 수정한 후
- 새 전략을 추가한 후

## Related Files

| File | Purpose |
|------|---------|
| `strategies/_common/trainer.py` | Walk-Forward 학습 + 모델 저장 (FIXED_PARAMS, atomic write) |
| `strategies/_common/features.py` | FeatureEngine — 피처 계산 + 선별 + 상관관계 제거 |
| `strategies/_common/labeler.py` | TripleBarrierLabeler — 모멘텀 전략용 라벨링 |
| `strategies/_common/evaluator.py` | ModelEvaluator — 과적합 체크, ML/트레이딩 메트릭 |
| `strategies/btc_1h_momentum/strategy.py` | BTC 모멘텀 전략 (LGBMClassifierStrategy) |
| `strategies/btc_1h_momentum/config.yaml` | BTC 모멘텀 전략 설정 |
| `strategies/btc_1h_momentum/models/` | BTC 모멘텀 학습된 모델 파일 |
| `strategies/eth_1h_momentum/strategy.py` | ETH 모멘텀 전략 (LGBMClassifierStrategy) |
| `strategies/eth_1h_momentum/config.yaml` | ETH 모멘텀 전략 설정 |
| `strategies/eth_1h_momentum/models/` | ETH 모멘텀 학습된 모델 파일 |
| `strategies/btc_1h_mean_reversion/strategy.py` | BTC 평균회귀 전략 (MeanReversionStrategy) |
| `strategies/btc_1h_mean_reversion/config.yaml` | BTC 평균회귀 전략 설정 |
| `strategies/btc_1h_mean_reversion/labeler.py` | MeanReversionLabeler — 평균회귀 전용 라벨러 |
| `strategies/btc_1h_mean_reversion/models/` | BTC 평균회귀 학습된 모델 파일 |
| `train_lgbm.py` | 학습 CLI 스크립트 (--strategy 인자로 전략 지정) |
| `oos_validation.py` | OOS 검증 CLI 스크립트 (--strategy 인자) |
| `main.py` | 실거래 루프 (Live 재학습 금지 대상) |

## Workflow

### Step 1: 이진 분류 고정 검증

**파일:** `strategies/_common/trainer.py`

3클래스(매수/중립/매도)에서 2클래스(매수/비매수)로 전환 완료. LABEL_MAP은 삭제되었고, LightGBM binary 모드를 사용한다.

```
Grep pattern="objective|is_unbalance" path="strategies/_common/trainer.py" output_mode="content"
```

**검증:**
1. `FIXED_PARAMS["objective"]` == `"binary"` (multiclass가 아님)
2. `FIXED_PARAMS["is_unbalance"]` == `False` (라벨 52/48 거의 균형)
3. `FIXED_PARAMS`에 `num_class` 키가 **없음** (binary 모드에서는 불필요)
4. Optuna 목적함수에서 `f1_score(..., average="binary", pos_label=1)` 사용

**PASS:** objective == "binary", is_unbalance == False, num_class 없음
**FAIL:** multiclass 관련 설정 잔존 또는 is_unbalance == True

### Step 2: 모델 파일 구조 검증

**파일:** 각 전략의 `models/` 디렉토리

모든 활성 전략에 대해 반복 검증한다:
- `strategies/btc_1h_momentum/models/`
- `strategies/eth_1h_momentum/models/`
- `strategies/btc_1h_mean_reversion/models/`

```
Glob pattern="strategies/btc_1h_momentum/models/*"
Glob pattern="strategies/eth_1h_momentum/models/*"
Glob pattern="strategies/btc_1h_mean_reversion/models/*"
```

각 전략의 models/ 디렉토리에 필수 4개 파일 존재 확인:
- `latest.txt` — LightGBM Booster 텍스트 포맷
- `feature_names.json` — 피처 이름 목록
- `best_params.json` — Optuna 최적 하이퍼파라미터
- `training_meta.json` — fold별 성과, 피처 importance, best_fold_idx

추가로 앙상블 전략은 config.yaml의 `ensemble_folds`에 지정된 `fold_XX.txt` 파일이 존재해야 한다.

각 전략의 strategy.py 내 기본 경로가 해당 전략 디렉토리를 가리키는지 확인:

```
Grep pattern="models/(latest\.txt|feature_names\.json)" path="strategies/btc_1h_momentum/strategy.py" output_mode="content"
Grep pattern="models/(latest\.txt|feature_names\.json)" path="strategies/eth_1h_momentum/strategy.py" output_mode="content"
Grep pattern="models/(latest\.txt|feature_names\.json)" path="strategies/btc_1h_mean_reversion/strategy.py" output_mode="content"
```

**PASS:** 필수 4개 파일 존재 + strategy.py 기본 경로가 해당 전략의 models/ 디렉토리와 일치
**FAIL:** 파일 누락 또는 경로가 다른 전략의 models/를 가리킴
**예외:** models/ 디렉토리가 비어있으면 "학습 필요" 안내 (위반 아님)

### Step 3: 피처 일관성 검증

**파일:** 각 전략의 `strategy.py`, `train_lgbm.py`

학습과 추론이 다른 피처 계산 로직을 사용하면 모델이 엉뚱한 입력을 받게 된다.

```
Grep pattern="from strategies._common.features import FeatureEngine" path="strategies/btc_1h_momentum/strategy.py" output_mode="content"
Grep pattern="from strategies._common.features import FeatureEngine" path="strategies/eth_1h_momentum/strategy.py" output_mode="content"
Grep pattern="from strategies._common.features import FeatureEngine" path="strategies/btc_1h_mean_reversion/strategy.py" output_mode="content"
Grep pattern="from strategies._common.features import FeatureEngine" path="train_lgbm.py" output_mode="content"
```

**검증:**
1. 모든 전략의 strategy.py와 train_lgbm.py가 **동일한** `strategies._common.features.FeatureEngine`을 import
2. 각 strategy.py가 `feature_names.json`에서 로드한 피처 이름으로 컬럼을 선택 (하드코딩 금지)
3. FeatureEngine 초기화 시 `config` 딕셔너리에서 `symbol` 키를 전달하여 심볼별 펀딩비 경로가 자동 결정되는지 확인

```
Grep pattern="FeatureEngine\(config" path="strategies/btc_1h_momentum/strategy.py" output_mode="content"
Grep pattern="FeatureEngine\(config" path="strategies/btc_1h_mean_reversion/strategy.py" output_mode="content"
Grep pattern='FeatureEngine\(config=\{"symbol"' path="train_lgbm.py" output_mode="content"
```

**PASS:** 동일 FeatureEngine + JSON 기반 피처 선택 + symbol 매개변수 전달
**FAIL:** 다른 피처 모듈 사용, 피처 이름 하드코딩, 또는 symbol 미전달

### Step 4: Walk-Forward 규칙 검증

**파일:** `strategies/_common/trainer.py`

```
Grep pattern="embargo_bars" path="strategies/_common/trainer.py" output_mode="content"
Grep pattern="min_train_months" path="strategies/_common/trainer.py" output_mode="content"
```

**검증 1: embargo_bars 기본값 ≥ 24** (CLAUDE.md 규칙: 24봉)
- `__init__`의 `embargo_bars` 기본값이 24 이상인지 확인
- `generate_folds()`에서 `train_idx[:-self.embargo_bars]`로 embargo가 실제 적용되는지 확인

**검증 2: min_train_months 기본값 ≥ 6** (CLAUDE.md 규칙: 최소 6개월)
- `__init__`의 `min_train_months` 기본값이 6 이상인지 확인

**검증 3: Optuna 첫 fold 전용**

`run()` 메서드를 Read로 읽고 다음 구체적 패턴을 확인:
- `optimize_hyperparams()` 호출이 **for 루프 밖**, 첫 fold 데이터(folds[0])에 대해서만 실행
- for 루프 안에서는 `train_fold()`만 호출하고, `optimize_hyperparams()`는 호출하지 않음
- `n_optuna_trials > 0` 조건으로 Optuna 비활성화 옵션이 존재

현재 올바른 패턴:
```python
# for 루프 밖 (첫 fold)
if self.n_optuna_trials > 0:
    best_params = self.optimize_hyperparams(X_train_0, y_train_0, X_val_0, y_val_0)

# for 루프 안 (모든 fold)
for i, fold in enumerate(folds):
    model, metrics = self.train_fold(..., best_params)  # 동일 파라미터 사용
```

만약 `optimize_hyperparams()`가 for 루프 내부에서 호출되면 FAIL이다 — 매 fold마다 재튜닝하면 과적합 위험이 급증한다.

**PASS:** embargo ≥ 24, min_train ≥ 6, Optuna 첫 fold 전용
**FAIL:** 위 조건 위반

### Step 5: Live 재학습 금지 검증

**파일:** `main.py`

```
Grep pattern="(WalkForwardTrainer|\.fit\(|\.train\(|train_lgbm|labeler|retrain|import.*trainer)" path="main.py" output_mode="content"
```

`main.py` 전체에서 다음이 발견되면 FAIL:
- `WalkForwardTrainer`, `TripleBarrierLabeler`, `MeanReversionLabeler` import 또는 사용
- `model.fit()`, `model.train()` 호출
- `train_lgbm` 모듈 import
- `retrain` 키워드

단, `load_strategy()` 내에서 `LGBMClassifierStrategy`, `MeanReversionStrategy`를 import하는 것은 허용 — 이것은 추론 전용이다.

**PASS:** main.py에 학습 관련 코드 없음
**FAIL:** 학습/재학습 코드 발견

### Step 6: Atomic Write 패턴 검증 (모델 저장)

**파일:** `strategies/_common/trainer.py`

```
Grep pattern="(tempfile|shutil\.move|_atomic_write)" path="strategies/_common/trainer.py" output_mode="content"
```

`save_model()` 메서드가 모든 파일 저장(latest.txt, feature_names.json, best_params.json, training_meta.json, fold_XX.txt)에 atomic write를 사용하는지 확인한다.

올바른 패턴:
- `_atomic_write_text()` 또는 `_atomic_write_json()` 헬퍼를 통해 저장
- 각 헬퍼는 `tempfile.mkstemp() → write → shutil.move()` 순서
- fold 모델도 `_atomic_write_text()`로 저장

직접 `open(path, 'w') → write()`로 저장하면 쓰기 중 크래시 시 파일이 손상된다.

**PASS:** 모든 모델 파일(latest.txt + fold_XX.txt + JSON 파일)이 atomic write 사용
**FAIL:** 직접 open → write 패턴 존재

### Step 7: config.yaml ML 파라미터 검증

**파일:** 각 전략의 `config.yaml`과 `strategy.py`

모든 활성 전략에 대해 반복 검증한다:

```
Read file_path="strategies/btc_1h_momentum/config.yaml"
Read file_path="strategies/eth_1h_momentum/config.yaml"
Read file_path="strategies/btc_1h_mean_reversion/config.yaml"
```

**검증 (각 전략별):**
1. `params.confidence_threshold`가 0 초과 1 이하 범위인지 (0이면 모든 예측을 신호로 변환, 1 초과면 신호 생성 불가)
2. `params.model_path`가 해당 전략의 `strategy.py` 내 `_load_model()` 기본 경로와 일치하는지
3. `params.feature_names_path`가 해당 전략의 `strategy.py` 내 `_load_feature_names()` 기본 경로와 일치하는지
4. `strategy.symbol`과 `strategy.timeframe`이 존재하는지
5. `params.ensemble_folds`가 설정된 경우, 해당 fold 번호의 `fold_XX.txt` 파일이 models/에 존재하는지
6. `params.models_dir`이 해당 전략의 models/ 디렉토리를 가리키는지
7. `params.funding_filter` 설정이 있는 경우, `zscore_thresholds` 리스트의 각 항목에 `zscore_below`와 `confidence` 키가 존재하는지
8. (btc_1h_mean_reversion만) `params.oi_filter` 설정이 있는 경우, `block_zscore` 값이 존재하는지

**PASS:** 모든 파라미터가 유효 범위 + 경로 정합
**FAIL:** 범위 초과, 경로 불일치, 또는 앙상블 fold 파일 누락

### Step 8: generate_signal ↔ generate_signals_vectorized 일관성 검증

**파일:** 각 전략의 `strategy.py`

각 전략의 strategy.py를 Read로 읽고 두 메서드의 로직을 비교한다.

**동치 조건 (모멘텀 전략 — btc_1h_momentum, eth_1h_momentum):**
1. 두 메서드 모두 `self.feature_engine.compute_all_features(df)` 호출
2. 두 메서드 모두 `self.feature_names`로 동일한 피처 컬럼 선택
3. 두 메서드 모두 NaN 행에 대해 0(비매수) 반환
4. 두 메서드 모두 `self._predict()` → 매수 확률 반환 → threshold 비교 → 1 또는 0 반환 (Booster.predict()는 binary에서 확률 스칼라 반환)
5. 펀딩비 적응형 threshold 로직이 동치 (generate_signal은 `_get_adaptive_threshold()`, vectorized는 인라인 배열 연산)
6. confidence_threshold 미만일 때 두 메서드 모두 0 반환

**추가 동치 조건 (평균회귀 전략 — btc_1h_mean_reversion):**
7. OI 필터 로직이 동치: generate_signal은 `oi_zscore >= block_zscore`이면 0 반환, vectorized는 `adaptive_thr[block_mask] = 999.0`으로 차단

두 메서드의 결과가 달라지면 실거래(generate_signal)와 백테스트(generate_signals_vectorized)가 다른 신호를 생성하게 되어 백테스트 신뢰성이 무너진다.

**PASS:** 두 메서드가 동치 로직 (펀딩비 필터 + OI 필터 포함)
**FAIL:** 피처 선택, 임계값, 필터 등에서 차이 발견

### Step 9: 과적합 체크 통합 검증

**파일:** `train_lgbm.py`, `strategies/_common/evaluator.py`

```
Grep pattern="check_overfitting" path="train_lgbm.py" output_mode="content"
Grep pattern="ModelEvaluator" path="train_lgbm.py" output_mode="content"
```

**검증:**
1. train_lgbm.py가 `ModelEvaluator.check_overfitting()`을 import하고 호출하는지
2. 호출 시 각 fold의 `train_f1_macro`와 `val_f1_macro`를 전달하는지
3. 과적합 감지 시 경고 로그를 출력하는지
4. 전체 fold가 과적합일 때 재검토 경고가 출력되는지

과적합 체크 없이 모델을 저장하면, 학습 데이터에만 잘 맞는 모델이 실거래에 투입될 위험이 있다.

**PASS:** check_overfitting이 모든 fold에 대해 호출 + 경고 로그 존재
**FAIL:** check_overfitting 미호출 또는 경고 누락

### Step 10: 피처 선별 파이프라인 일관성 검증

**파일:** `strategies/_common/features.py`, `train_lgbm.py`, 각 전략의 `strategy.py`

```
Grep pattern="get_selected_features|remove_correlated_features" path="strategies/_common/features.py" output_mode="content"
Grep pattern="get_selected_features|remove_correlated_features|use.all.features" path="train_lgbm.py" output_mode="content"
```

**검증:**
1. `FeatureEngine.get_selected_features()`가 존재하고, 선별된 피처 이름 리스트를 반환하는지
2. `FeatureEngine.get_feature_names()`가 전체 피처 이름을 반환하는지
3. `train_lgbm.py`에서 `--use-all-features` 플래그에 따라 `get_feature_names()` 또는 `get_selected_features()` 분기가 존재하는지
4. 상관관계 제거: `FeatureEngine.remove_correlated_features()`가 존재하고 train_lgbm.py에서 `--corr-threshold`에 따라 호출되는지
5. 학습 시 사용된 피처 이름이 `feature_names.json`에 저장되는지 (Step 2 save_model과 연계)
6. 각 strategy.py는 `feature_names.json`에서 로드한 피처만 사용하므로 학습/추론 일관성이 JSON 파일을 통해 보장되는지

**핵심 원리:** 학습 시 어떤 피처를 선별하든 `feature_names.json`에 저장하면 strategy.py가 동일 피처를 사용한다. 하지만 `FeatureEngine.compute_all_features()`가 해당 피처를 생성하지 않으면 KeyError가 발생한다.

**PASS:** 학습 시 선별된 피처가 feature_names.json에 저장되고, strategy.py가 이를 로드하여 사용
**FAIL:** 피처 선별 후 feature_names.json 미저장, 또는 compute_all_features()에 없는 피처가 포함

### Step 11: Best Fold 선택 로직 검증

**파일:** `strategies/_common/trainer.py`

**도구:** Read

trainer.py의 `run()` 메서드 마지막 부분에서 best fold 선택 로직을 읽고 확인합니다:

```
Grep pattern="eligible|selected_fold|MIN_VAL_F1|max_overfit_gap" path="strategies/_common/trainer.py" output_mode="content"
```

**검증:**
1. 각 fold의 `overfit_gap = train_f1_macro - val_f1_macro`이 계산되는지
2. 유효 fold 필터링: `train_f1_macro >= MIN_TRAIN_F1 (0.01)` AND `overfit_gap <= max_overfit_gap` AND `val_f1_macro >= MIN_VAL_F1 (0.40)`
3. 유효 fold 중 **최신 fold 우선** 선택 (`max(eligible, key=lambda x: (x[0], x[1]))`)
4. 유효 fold가 없을 때 fallback: 학습 성공 fold 중 gap 최소 → 경고 출력
5. `save_model()`에 `best_fold_idx`와 `best_val_f1`이 전달되는지
6. `training_meta.json`에 `best_fold_idx`와 `best_val_f1`이 저장되는지

**PASS:** best fold 선택이 eligible 필터 + 최신 fold 우선으로 올바르게 구현
**FAIL:** 단순히 마지막 fold만 사용하거나, eligible 필터 누락

### Step 12: labeler_type 동적 선택 검증

**파일:** `train_lgbm.py`, `strategies/btc_1h_mean_reversion/labeler.py`, `strategies/_common/labeler.py`

```
Grep pattern="labeler_type|MeanReversionLabeler|TripleBarrierLabeler" path="train_lgbm.py" output_mode="content"
```

**검증:**
1. `train_lgbm.py`에서 전략 config.yaml의 `params.labeler_type` 값을 읽는지
2. `labeler_type == "mean_reversion"`이면 `strategies.btc_1h_mean_reversion.labeler.MeanReversionLabeler`를 import하여 사용하는지
3. 기본값(미지정 또는 `"triple_barrier"`)이면 `strategies._common.labeler.TripleBarrierLabeler`를 사용하는지
4. MeanReversionLabeler 초기화 시 config.yaml의 파라미터(oversold_bb_threshold, oversold_rsi_threshold, oversold_mode 등)를 전달하는지
5. 두 라벨러 모두 `generate_labels(df)` 메서드로 2클래스 라벨(0, 1)을 반환하는지 (3클래스 -1/0/1이 아님)

```
Grep pattern="def generate_labels" path="strategies/_common/labeler.py" output_mode="content"
Grep pattern="def generate_labels" path="strategies/btc_1h_mean_reversion/labeler.py" output_mode="content"
```

**PASS:** labeler_type에 따라 올바른 라벨러가 선택되고, 2클래스 라벨을 반환
**FAIL:** labeler_type 분기 누락, 3클래스 라벨 반환, 또는 라벨러 파라미터 미전달

## Output Format

```markdown
### verify-ml 검증 결과

| # | 검사 항목 | 대상 | 결과 | 상세 |
|---|-----------|------|------|------|
| 1 | 이진 분류 고정 | trainer.py FIXED_PARAMS | PASS/FAIL | ... |
| 2 | 모델 파일 구조 | 각 전략 models/ | PASS/FAIL | ... |
| 3 | 피처 일관성 | strategy.py ↔ train_lgbm.py | PASS/FAIL | ... |
| 4 | Walk-Forward 규칙 | trainer.py | PASS/FAIL | ... |
| 5 | Live 재학습 금지 | main.py | PASS/FAIL | ... |
| 6 | Atomic Write (모델) | trainer.py | PASS/FAIL | ... |
| 7 | config.yaml ML 파라미터 | 각 전략 config.yaml | PASS/FAIL | ... |
| 8 | signal ↔ vectorized 일관성 | 각 전략 strategy.py | PASS/FAIL | ... |
| 9 | 과적합 체크 통합 | train_lgbm.py + evaluator.py | PASS/FAIL | ... |
| 10 | 피처 선별 파이프라인 | features.py + train_lgbm.py | PASS/FAIL | ... |
| 11 | Best Fold 선택 로직 | trainer.py | PASS/FAIL | ... |
| 12 | labeler_type 동적 선택 | train_lgbm.py + labeler.py | PASS/FAIL | ... |
```

## Exceptions

1. **models/ 디렉토리 비어있음 (학습 전)** — 아직 train_lgbm.py를 실행하지 않은 경우 모델 파일이 없는 것은 위반이 아님. "학습 필요" 안내로 대체.
2. **테스트 코드의 학습 호출** — `tests/` 디렉토리 내에서 WalkForwardTrainer, FeatureEngine, fit()을 사용하는 것은 허용 (테스트 목적)
3. **config.yaml의 모델 경로 오버라이드** — config.yaml에서 model_path/feature_names_path를 커스텀 경로로 지정하는 것은 허용되지만, 해당 경로에 파일이 실제로 존재하는지는 확인 필요
4. **evaluator.py의 메트릭 변경** — ML 메트릭(F1, AUC 등) 추가/제거는 분류 체계 변경이 아니므로 위반이 아님
5. **generate_signals_vectorized 미구현** — BaseStrategy에서 필수가 아니므로 미구현 자체는 위반이 아님. 단, 구현되어 있다면 generate_signal과 로직이 동치여야 함
6. **새 전략 추가 시** — 기존 전략과 동일한 패턴(BaseStrategy 상속, FeatureEngine 사용, config.yaml 구조)을 따르면 위반이 아님. verify-strategy 스킬로 별도 검증.
