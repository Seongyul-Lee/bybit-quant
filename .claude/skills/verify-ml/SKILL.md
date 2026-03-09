---
name: verify-ml
description: ML 전략(LightGBM) 전용 규칙 준수 여부를 검증합니다. ML 전략 코드 수정, 학습 파이프라인 변경, 모델 파일 구조 변경 후 사용. strategies/lgbm_classifier/ 내 파일, train_lgbm.py, 또는 ML 관련 config 수정 시 반드시 실행하세요.
---

# ML 전략 규칙 검증

## 검증 항목 요약

| # | 검사 항목 | 핵심 목적 |
|---|-----------|-----------|
| 1 | 라벨 매핑 삼중 동기화 | trainer/strategy/labeler 간 라벨 매핑이 완전한 일관성을 유지 |
| 2 | 모델 파일 구조 | 필수 4개 파일 존재 + strategy.py 참조 경로 일치 |
| 3 | 피처 일관성 | 학습/추론이 동일한 FeatureEngine을 사용 |
| 4 | Walk-Forward 규칙 | embargo, 최소 학습 기간, Optuna 첫 fold 전용 |
| 5 | Live 재학습 금지 | main.py run_live()에 학습 코드 부재 |
| 6 | 3클래스 분류 고정 | num_class=3, _n_classes=3 일관 |
| 7 | Atomic Write (모델) | 모델 저장 시 tempfile → shutil.move 패턴 |
| 8 | config.yaml ML 파라미터 | confidence_threshold 범위, 경로 정합성 |
| 9 | generate_signal ↔ vectorized 일관성 | 두 메서드의 로직 동치 |
| 10 | 과적합 체크 통합 | train_lgbm.py에서 ModelEvaluator.check_overfitting 호출 |

## When to Run

- `strategies/lgbm_classifier/` 내 Python 파일을 수정한 후
- `train_lgbm.py`를 수정한 후
- ML 모델 저장/로드 로직을 변경한 후
- 피처 엔지니어링(features.py)을 수정한 후
- 라벨링(labeler.py) 로직을 변경한 후
- `strategies/lgbm_classifier/config.yaml`을 수정한 후

## Related Files

| File | Purpose |
|------|---------|
| `strategies/lgbm_classifier/strategy.py` | ML 전략 구현 (추론, 라벨 역매핑) |
| `strategies/lgbm_classifier/trainer.py` | Walk-Forward 학습 + 모델 저장 (라벨 매핑) |
| `strategies/lgbm_classifier/features.py` | FeatureEngine — 피처 계산 (~50개) |
| `strategies/lgbm_classifier/labeler.py` | Triple Barrier 라벨링 |
| `strategies/lgbm_classifier/evaluator.py` | 모델 평가 (ML + 트레이딩 메트릭) |
| `strategies/lgbm_classifier/config.yaml` | ML 전략 설정 |
| `train_lgbm.py` | 학습 CLI 스크립트 |
| `main.py` | 실거래 루프 (Live 재학습 금지 대상) |

## Workflow

### Step 1: 라벨 매핑 삼중 동기화 검증

**파일:** `trainer.py`, `strategy.py`, `labeler.py`

라벨 매핑은 3곳에 분산되어 있어 하나라도 어긋나면 학습-추론 불일치가 발생한다.

```
Grep pattern="LABEL_MAP" path="strategies/lgbm_classifier/trainer.py" output_mode="content"
Grep pattern="_LABEL_MAP_INV" path="strategies/lgbm_classifier/strategy.py" output_mode="content"
```

**검증할 3가지 매핑:**
1. **trainer.py `LABEL_MAP`** — 원본(-1,0,1) → 학습용(0,1,2) 변환
2. **trainer.py `LABEL_MAP_INV`** — 학습용(0,1,2) → 원본(-1,0,1) 역변환
3. **strategy.py `_LABEL_MAP_INV`** — 추론 시 학습용(0,1,2) → 원본(-1,0,1) 역변환

**일관성 조건:**
- `LABEL_MAP`과 `LABEL_MAP_INV`가 완전한 역함수: 모든 k,v에 대해 `LABEL_MAP[k]=v ↔ LABEL_MAP_INV[v]=k`
- `trainer.py LABEL_MAP_INV`와 `strategy.py _LABEL_MAP_INV`의 값이 동일
- 매핑 키 집합이 정확히 3개 (3클래스)

**추가로, labeler 라벨값과 LABEL_MAP 키 일치 확인:**

```
Grep pattern="label\s*=\s*(-?[01])" path="strategies/lgbm_classifier/labeler.py" output_mode="content"
```

labeler.py의 `generate_labels()`가 반환하는 값 집합 {-1, 0, 1}이 LABEL_MAP의 키 집합 {-1, 0, 1}과 정확히 일치하는지 확인한다. labeler가 새로운 라벨값(예: 2)을 생성하면 LABEL_MAP에 매핑이 없어 KeyError가 발생한다.

**PASS:** 세 매핑이 완전히 일관되고, labeler 라벨값이 LABEL_MAP 키와 일치
**FAIL:** 매핑 불일치, 키-값 쌍 누락, 또는 labeler 라벨값이 LABEL_MAP에 없음

### Step 2: 모델 파일 구조 검증

**파일:** `strategies/lgbm_classifier/models/`

```
Glob pattern="strategies/lgbm_classifier/models/*"
```

필수 파일 4개 존재 확인:
- `latest.txt` — LightGBM Booster 텍스트 포맷
- `feature_names.json` — 피처 이름 목록
- `best_params.json` — Optuna 최적 하이퍼파라미터
- `training_meta.json` — fold별 성과, 피처 importance

strategy.py의 로드 코드가 참조하는 기본 경로가 위 파일과 일치하는지 확인:

```
Grep pattern="models/(latest\.txt|feature_names\.json)" path="strategies/lgbm_classifier/strategy.py" output_mode="content"
```

**PASS:** 필수 4개 파일 존재 + strategy.py 기본 경로 일치
**FAIL:** 파일 누락 또는 경로 불일치
**예외:** models/ 디렉토리가 비어있으면 "학습 필요" 안내 (위반 아님)

### Step 3: 피처 일관성 검증

**파일:** `strategy.py`, `train_lgbm.py`

학습과 추론이 다른 피처 계산 로직을 사용하면 모델이 엉뚱한 입력을 받게 된다.

```
Grep pattern="from strategies.lgbm_classifier.features import FeatureEngine" path="strategies/lgbm_classifier/strategy.py" output_mode="content"
Grep pattern="from strategies.lgbm_classifier.features import FeatureEngine" path="train_lgbm.py" output_mode="content"
```

**검증:**
1. 두 파일이 **동일한** `strategies.lgbm_classifier.features.FeatureEngine`을 import
2. strategy.py가 `feature_names.json`에서 로드한 피처 이름으로 컬럼을 선택 (하드코딩 금지)

```
Grep pattern="feature_names" path="strategies/lgbm_classifier/strategy.py" output_mode="content"
```

strategy.py의 `_load_feature_names()`가 JSON 파일에서 피처 이름을 읽고, `generate_signal()`과 `generate_signals_vectorized()`에서 `self.feature_names`로 컬럼을 선택하는지 확인한다.

**PASS:** 동일 FeatureEngine + JSON 기반 피처 선택
**FAIL:** 다른 피처 모듈 사용 또는 피처 이름 하드코딩

### Step 4: Walk-Forward 규칙 검증

**파일:** `strategies/lgbm_classifier/trainer.py`

```
Grep pattern="embargo_bars" path="strategies/lgbm_classifier/trainer.py" output_mode="content"
Grep pattern="min_train_months" path="strategies/lgbm_classifier/trainer.py" output_mode="content"
```

**검증 1: embargo_bars 기본값 > 0** (CLAUDE.md 규칙: 24봉)
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
Grep pattern="(WalkForwardTrainer|FeatureEngine|\.fit\(|\.train\(|train_lgbm|labeler|retrain|import.*trainer|import.*features)" path="main.py" output_mode="content"
```

`run_live()` 함수 내에서 다음이 발견되면 FAIL:
- `WalkForwardTrainer`, `FeatureEngine`, `TripleBarrierLabeler` import 또는 사용
- `model.fit()`, `model.train()` 호출
- `train_lgbm` 모듈 import
- `retrain` 키워드

단, `load_strategy()` 내에서 `LGBMClassifierStrategy`를 import하는 것은 허용 — 이것은 추론 전용이다.

**PASS:** run_live()에 학습 관련 코드 없음
**FAIL:** 학습/재학습 코드 발견

### Step 6: 3클래스 분류 고정 검증

**파일:** `trainer.py`, `strategy.py`

```
Grep pattern="num_class" path="strategies/lgbm_classifier/trainer.py" output_mode="content"
Grep pattern="_n_classes" path="strategies/lgbm_classifier/strategy.py" output_mode="content"
```

**검증:**
- `FIXED_PARAMS["num_class"]` = 3
- `model._n_classes` = 3 (strategy.py `_load_model()`에서 설정)
- `_LABEL_MAP_INV`의 키가 정확히 {0, 1, 2} (3클래스)

세 값이 모두 3으로 일관되어야 한다. 향후 클래스 수를 변경하려면 이 세 곳을 동시에 수정해야 하므로 불일치 방지가 중요하다.

**PASS:** 3클래스 일관
**FAIL:** num_class, _n_classes, LABEL_MAP_INV 키 수 불일치

### Step 7: Atomic Write 패턴 검증 (모델 저장)

**파일:** `strategies/lgbm_classifier/trainer.py`

```
Grep pattern="(tempfile|shutil\.move|_atomic_write)" path="strategies/lgbm_classifier/trainer.py" output_mode="content"
```

`save_model()` 메서드가 모든 파일 저장(latest.txt, feature_names.json, best_params.json, training_meta.json)에 atomic write를 사용하는지 확인한다.

올바른 패턴:
- `_atomic_write_text()` 또는 `_atomic_write_json()` 헬퍼를 통해 저장
- 각 헬퍼는 `tempfile.mkstemp() → write → shutil.move()` 순서

직접 `open(path, 'w') → write()`로 저장하면 쓰기 중 크래시 시 파일이 손상된다.

**PASS:** 모든 모델 파일이 atomic write 사용
**FAIL:** 직접 open → write 패턴 존재

### Step 8: config.yaml ML 파라미터 검증

**파일:** `strategies/lgbm_classifier/config.yaml`, `strategy.py`

```
Read file_path="strategies/lgbm_classifier/config.yaml"
```

**검증:**
1. `params.confidence_threshold`가 0 초과 1 이하 범위인지 (0이면 모든 예측을 신호로 변환, 1 초과면 신호 생성 불가)
2. `params.model_path`가 strategy.py의 `_load_model()` 기본값(`strategies/lgbm_classifier/models/latest.txt`)과 일치하는지
3. `params.feature_names_path`가 strategy.py의 `_load_feature_names()` 기본값(`strategies/lgbm_classifier/models/feature_names.json`)과 일치하는지
4. `strategy.name`이 `LGBMClassifierStrategy`인지 (main.py의 load_strategy 분기와 일치)
5. `strategy.symbol`과 `strategy.timeframe`이 존재하는지

**PASS:** 모든 파라미터가 유효 범위 + 경로 정합
**FAIL:** 범위 초과 또는 경로 불일치

### Step 9: generate_signal ↔ generate_signals_vectorized 일관성 검증

**파일:** `strategies/lgbm_classifier/strategy.py`

strategy.py를 Read로 읽고 두 메서드의 로직을 비교한다.

**동치 조건:**
1. 두 메서드 모두 `self.feature_engine.compute_all_features(df)` 호출
2. 두 메서드 모두 `self.feature_names`로 동일한 피처 컬럼 선택
3. 두 메서드 모두 NaN 행에 대해 0(중립) 반환
4. 두 메서드 모두 `self.model.predict_proba()` → `argmax` → `self.confidence_threshold` 비교 → `self._LABEL_MAP_INV` 역매핑 동일 순서
5. confidence_threshold 미만일 때 두 메서드 모두 0 반환

두 메서드의 결과가 달라지면 실거래(generate_signal)와 백테스트(generate_signals_vectorized)가 다른 신호를 생성하게 되어 백테스트 신뢰성이 무너진다.

**PASS:** 두 메서드가 동치 로직
**FAIL:** 피처 선택, 임계값, 역매핑 등에서 차이 발견

### Step 10: 과적합 체크 통합 검증

**파일:** `train_lgbm.py`, `strategies/lgbm_classifier/evaluator.py`

```
Grep pattern="check_overfitting" path="train_lgbm.py" output_mode="content"
Grep pattern="ModelEvaluator" path="train_lgbm.py" output_mode="content"
```

**검증:**
1. train_lgbm.py가 `ModelEvaluator.check_overfitting()`을 import하고 호출하는지
2. 호출 시 각 fold의 train_f1과 val_f1을 전달하는지
3. 과적합 감지 시 경고 로그를 출력하는지

과적합 체크 없이 모델을 저장하면, 학습 데이터에만 잘 맞는 모델이 실거래에 투입될 위험이 있다.

**PASS:** check_overfitting이 모든 fold에 대해 호출 + 경고 로그 존재
**FAIL:** check_overfitting 미호출 또는 경고 누락

## Output Format

```markdown
### verify-ml 검증 결과

| # | 검사 항목 | 대상 | 결과 | 상세 |
|---|-----------|------|------|------|
| 1 | 라벨 매핑 삼중 동기화 | trainer.py ↔ strategy.py ↔ labeler.py | PASS/FAIL | ... |
| 2 | 모델 파일 구조 | models/ 디렉토리 | PASS/FAIL | ... |
| 3 | 피처 일관성 | strategy.py ↔ train_lgbm.py | PASS/FAIL | ... |
| 4 | Walk-Forward 규칙 | trainer.py | PASS/FAIL | ... |
| 5 | Live 재학습 금지 | main.py | PASS/FAIL | ... |
| 6 | 3클래스 분류 고정 | trainer.py + strategy.py | PASS/FAIL | ... |
| 7 | Atomic Write (모델) | trainer.py | PASS/FAIL | ... |
| 8 | config.yaml ML 파라미터 | config.yaml + strategy.py | PASS/FAIL | ... |
| 9 | signal ↔ vectorized 일관성 | strategy.py | PASS/FAIL | ... |
| 10 | 과적합 체크 통합 | train_lgbm.py + evaluator.py | PASS/FAIL | ... |
```

## Exceptions

1. **models/ 디렉토리 비어있음 (학습 전)** — 아직 train_lgbm.py를 실행하지 않은 경우 모델 파일이 없는 것은 위반이 아님. "학습 필요" 안내로 대체.
2. **테스트 코드의 학습 호출** — `tests/` 디렉토리 내에서 WalkForwardTrainer, FeatureEngine, fit()을 사용하는 것은 허용 (테스트 목적)
3. **config.yaml의 모델 경로 오버라이드** — config.yaml에서 model_path/feature_names_path를 커스텀 경로로 지정하는 것은 허용되지만, 해당 경로에 파일이 실제로 존재하는지는 확인 필요
4. **evaluator.py의 메트릭 변경** — ML 메트릭(F1, AUC 등) 추가/제거는 분류 체계 변경이 아니므로 위반이 아님
5. **generate_signals_vectorized 미구현** — BaseStrategy에서 필수가 아니므로 미구현 자체는 위반이 아님. 단, 구현되어 있다면 generate_signal과 로직이 동치여야 함
