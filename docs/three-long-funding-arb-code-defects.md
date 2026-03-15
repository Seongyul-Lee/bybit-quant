# 3롱 + funding_arb 코드 결함 리뷰

## 범위
- 대상 전략
  - `btc_1h_momentum`
  - `eth_1h_momentum`
  - `btc_1h_mean_reversion`
  - `funding_arb` 라이브 경로 (BTC/ETH)
- 기준
  - `README.md` 제외
  - 실제 실행 경로, import chain, 설정 반영 여부 기준
  - 정적 코드 추적으로 확인한 결함만 기재

## 요약
현재 기본 라이브 경로는 `main.py`에서 3롱과 `funding_arb`를 함께 기동하지만, 포트폴리오 배분/리스크 스케일링이 실제 주문 수량에 반영되지 않고, 동일 심볼 BTC 전략 간 병행도 막혀 있으며, `funding_arb`와 3롱이 동일 선물 포지션을 공유해 서로를 상쇄할 수 있다. 추가로 `eth_1h_momentum`은 ETH funding feature 대신 BTC funding feature를 읽는 결함이 있다.

## 패치 우선순위

### P0. 즉시 수정 필요
1. `eth_1h_momentum`의 funding feature 심볼 오염 수정
2. `funding_arb`와 3롱의 동일 perp 심볼 충돌 해소
3. BTC 2전략(`btc_1h_momentum`, `btc_1h_mean_reversion`) 동시 운용 차단 로직 제거 또는 구조 재설계

### P1. 높은 우선순위
4. `PortfolioManager.allocate()`의 배분/스케일링 결과가 실제 주문 수량에 반영되도록 수정
5. `funding_arb`의 `capital_pct`가 코인별 중복 적용되는 문제 수정
6. `ArbRiskMonitor` 상태를 코인별로 분리

### P2. 중간 우선순위
7. `funding_arb` 리스크 설정 중 미사용 값과 dead path 정리
8. `funding_arb` 자동 청산/감축 정책을 코드로 연결

## 상세 결함

### 1. ETH 전략이 BTC funding feature를 읽음
- 우선순위: `P0`
- 영향
  - `eth_1h_momentum` 추론에 ETH 시장 외부의 BTC funding 상태가 섞인다.
  - 모델이 학습 시 기대한 입력과 실거래 입력이 불일치할 수 있다.
- 근거
  - 전략 인스턴스 생성 시 `config.yaml` 전체가 아니라 `config.get("params", {})`만 전달된다.  
    [src/portfolio/manager.py:376](/C:/Users/tjddb/bybit-quant/src/portfolio/manager.py#L376)
  - `FeatureEngine`은 `config["symbol"]`이 없으면 기본값 `BTCUSDT`를 사용한다.  
    [strategies/_common/features.py:38](/C:/Users/tjddb/bybit-quant/strategies/_common/features.py#L38)
  - funding feature 로딩 경로는 `self.symbol` 기반으로 생성된다.  
    [strategies/_common/features.py:367](/C:/Users/tjddb/bybit-quant/strategies/_common/features.py#L367)
  - ETH 전략은 실제로 `funding_rate_zscore`를 feature로 사용한다.  
    [strategies/eth_1h_momentum/models/feature_names.json](/C:/Users/tjddb/bybit-quant/strategies/eth_1h_momentum/models/feature_names.json)
- 원인
  - `strategy.symbol`이 `params`에 병합되지 않아 전략 객체/FeatureEngine이 심볼을 모른다.
- 권장 패치
  - 전략 생성 시 `strategy.symbol`, `strategy.timeframe`, `strategy.name`을 `params`에 명시적으로 병합해 전달.
  - 또는 `FeatureEngine`에 심볼을 별도 인자로 전달.

### 2. funding_arb와 3롱이 동일 선물 포지션을 공유해 서로 상쇄함
- 우선순위: `P0`
- 영향
  - `funding_arb` 숏과 v1 롱이 독립 sleeve로 공존하지 못한다.
  - v1 롱 주문이 arb short를 줄이거나 뒤집을 수 있다.
  - 결과적으로 "`3롱 + funding_arb` 동시 포트폴리오"가 구조적으로 깨진다.
- 근거
  - `funding_arb`는 기존 선물 실행기 `OrderExecutor`를 공유한다.  
    [main.py:467](/C:/Users/tjddb/bybit-quant/main.py#L467)  
    [main.py:490](/C:/Users/tjddb/bybit-quant/main.py#L490)
  - `funding_arb`는 동일 BTC/ETH perp 심볼에 숏을 연다.  
    [strategies/funding_arb/config.yaml:7](/C:/Users/tjddb/bybit-quant/strategies/funding_arb/config.yaml#L7)  
    [src/execution/arb_executor.py:79](/C:/Users/tjddb/bybit-quant/src/execution/arb_executor.py#L79)
  - v1 전략도 같은 BTC/ETH perp 심볼에 주문한다.  
    [main.py:613](/C:/Users/tjddb/bybit-quant/main.py#L613)  
    [main.py:839](/C:/Users/tjddb/bybit-quant/main.py#L839)
  - 실제 주문은 `VirtualPositionTracker.get_delta_orders()`가 현재 실제 포지션 대비 차이만큼 낸다.  
    [src/portfolio/virtual_position.py:123](/C:/Users/tjddb/bybit-quant/src/portfolio/virtual_position.py#L123)
  - arb 포지션은 `VirtualPositionTracker`에 등록되지 않는다.
- 원인
  - 3롱 포트폴리오와 arb 포지션이 서로 다른 논리로 관리되지만, 실제 선물 계정/심볼은 공유한다.
- 권장 패치
  - 방법 A: `funding_arb`를 별도 계정/서브계정/실행기 상태로 분리.
  - 방법 B: 하나의 포지션 추적 계층으로 통합해 arb leg도 virtual state에 포함.
  - 방법 C: 동일 심볼에 대해 v1과 arb를 동시에 허용하지 않는 하드 가드 추가.

### 3. BTC 2전략이 동시에 포지션을 가질 수 없음
- 우선순위: `P0`
- 영향
  - `btc_1h_momentum`과 `btc_1h_mean_reversion` 둘 다 active여도 먼저 진입한 하나만 유지되고, 나머지는 스킵된다.
  - 문서/설정상 3롱이어도 실거래에서는 사실상 `BTC 1개 + ETH 1개`처럼 동작할 수 있다.
- 근거
  - `VirtualPositionTracker`는 여러 전략의 동일 심볼 가상 포지션 합산을 지원하도록 설계돼 있다.  
    [src/portfolio/virtual_position.py:1](/C:/Users/tjddb/bybit-quant/src/portfolio/virtual_position.py#L1)
  - 하지만 실거래 루프는 이미 같은 방향 실제 포지션이 있으면 즉시 스킵한다.  
    [main.py:770](/C:/Users/tjddb/bybit-quant/main.py#L770)  
    [main.py:773](/C:/Users/tjddb/bybit-quant/main.py#L773)
- 원인
  - 가상 합산 구조를 도입해놓고, 실제 루프에서 심볼 단위 단일 포지션 가드가 더 강하게 적용된다.
- 권장 패치
  - `existing_pos["side"] == direction`이면 스킵하지 말고 가상 포지션을 열고 delta order를 계산하도록 수정.
  - 반대 방향 충돌만 별도 처리.

### 4. 포트폴리오 배분/스케일링이 실제 수량에 반영되지 않음
- 우선순위: `P1`
- 영향
  - `position_pct_per_strategy`
  - MDD 기반 `portfolio_scale`
  - 전략별 `strategy_scales`
  - 심볼/총노출 cap
  - 위 값들이 최종 주문 수량에 거의 반영되지 않는다.
- 근거
  - `PortfolioManager.allocate()`는 `effective_pct`와 cap 적용 후 주문 리스트를 만든다.  
    [src/portfolio/manager.py:208](/C:/Users/tjddb/bybit-quant/src/portfolio/manager.py#L208)  
    [src/portfolio/manager.py:234](/C:/Users/tjddb/bybit-quant/src/portfolio/manager.py#L234)
  - 그러나 실제 루프는 `order["size_pct"]`를 사용하지 않고 ATR 기반 수량을 새로 계산한다.  
    [main.py:748](/C:/Users/tjddb/bybit-quant/main.py#L748)  
    [main.py:803](/C:/Users/tjddb/bybit-quant/main.py#L803)
  - ATR 계산 결과는 `risk_params.yaml`의 5% 상한만 따른다.  
    [config/risk_params.yaml:1](/C:/Users/tjddb/bybit-quant/config/risk_params.yaml#L1)
- 원인
  - 포트폴리오 레이어와 리스크 레이어가 서로 다른 sizing 소스를 사용한다.
- 권장 패치
  - `allocate()` 결과의 `size_pct`를 실제 notional sizing 입력으로 사용.
  - `RiskManager.calculate_atr_position_size()`는 상한/조정 계수만 적용하는 보조 함수로 축소.

### 5. funding_arb의 capital_pct가 BTC/ETH 각각에 중복 적용됨
- 우선순위: `P1`
- 영향
  - 현재 설정 `capital_pct: 0.40`은 총 40%가 아니라 BTC 40% + ETH 40%로 사용된다.
  - 포트폴리오 전체 익스포저가 의도보다 커질 수 있다.
- 근거
  - 설정은 `portfolio.funding_arb.capital_pct: 0.40`.  
    [config/portfolio.yaml:54](/C:/Users/tjddb/bybit-quant/config/portfolio.yaml#L54)
  - `_run_funding_arb_check()`는 `symbols` 루프 안에서 매번 `capital_for_arb = portfolio_value * capital_pct`를 계산한다.  
    [main.py:243](/C:/Users/tjddb/bybit-quant/main.py#L243)  
    [main.py:262](/C:/Users/tjddb/bybit-quant/main.py#L262)
- 원인
  - `capital_pct`를 sleeve 총합 비중이 아니라 pair별 비중으로 적용.
- 권장 패치
  - `capital_pct / len(symbols)`로 pair별 배분.
  - 또는 설정명을 `capital_pct_per_pair`로 바꾸고 의미를 명확히 함.

### 6. ArbRiskMonitor가 BTC/ETH 상태를 분리하지 않음
- 우선순위: `P1`
- 영향
  - BTC, ETH의 funding 연속 음수 횟수와 누적 funding 상태가 서로 섞인다.
  - 잘못된 경고/긴급 알림이 발생할 수 있다.
- 근거
  - monitor 상태는 단일 전역 변수다.  
    [src/risk/arb_monitor.py:48](/C:/Users/tjddb/bybit-quant/src/risk/arb_monitor.py#L48)
  - `_check_funding_settlement()`는 코인별 루프에서 같은 monitor 인스턴스에 `check_funding_trend()`를 호출한다.  
    [main.py:401](/C:/Users/tjddb/bybit-quant/main.py#L401)  
    [main.py:412](/C:/Users/tjddb/bybit-quant/main.py#L412)
- 원인
  - strategy-level monitor를 pair-level state로 만들지 않았다.
- 권장 패치
  - `coin` 또는 `symbol_perp` 키로 상태를 분리한 dict 구조로 변경.

### 7. funding_arb 리스크 설정 일부가 dead config / dead code 상태
- 우선순위: `P2`
- 영향
  - 설정이 있어도 운영자가 기대한 보호가 작동하지 않는다.
- 근거
  - `close_position()`는 정의돼 있지만 호출 지점이 없다.  
    [src/execution/arb_executor.py:139](/C:/Users/tjddb/bybit-quant/src/execution/arb_executor.py#L139)
  - `max_cumulative_loss_pct`는 읽기만 하고 체크 로직이 없다.  
    [src/risk/arb_monitor.py:46](/C:/Users/tjddb/bybit-quant/src/risk/arb_monitor.py#L46)
  - `check_entry_slippage()`도 정의돼 있지만 실제 진입 후 호출되지 않는다.  
    [src/risk/arb_monitor.py:208](/C:/Users/tjddb/bybit-quant/src/risk/arb_monitor.py#L208)
  - `check_interval_sec`, `funding_schedule_utc`, `min_funding_rate`도 라이브 경로에서는 실질 반영되지 않는다.  
    [strategies/funding_arb/config.yaml:20](/C:/Users/tjddb/bybit-quant/strategies/funding_arb/config.yaml#L20)
- 권장 패치
  - 미사용 설정을 제거하거나 실제 enforcement path를 연결.
  - 자동 감축/청산 조건을 명시적으로 코드화.

## 권장 패치 순서

### 1단계. 데이터/모델 입력 정합성 복구
- 목표
  - 잘못된 입력으로 추론하는 문제부터 제거
- 작업
  - 전략 생성 시 `strategy.symbol`, `strategy.timeframe`, `strategy.name`을 `params`에 병합
  - `FeatureEngine` 테스트에 ETH 케이스 추가
- 대상
  - [src/portfolio/manager.py](/C:/Users/tjddb/bybit-quant/src/portfolio/manager.py)
  - [main.py](/C:/Users/tjddb/bybit-quant/main.py)
  - [portfolio_backtest.py](/C:/Users/tjddb/bybit-quant/portfolio_backtest.py)

### 2단계. 포지션 충돌 제거
- 목표
  - `funding_arb`와 3롱의 물리 포지션 간섭 방지
- 작업
  - arb를 별도 executor/계정으로 분리하거나
  - 공통 포지션 상태 계층으로 통합
  - 최소한 동일 심볼 동시 운용 금지 가드 추가
- 대상
  - [main.py](/C:/Users/tjddb/bybit-quant/main.py)
  - [src/execution/arb_executor.py](/C:/Users/tjddb/bybit-quant/src/execution/arb_executor.py)
  - [src/portfolio/virtual_position.py](/C:/Users/tjddb/bybit-quant/src/portfolio/virtual_position.py)

### 3단계. 3롱 구성 복원
- 목표
  - BTC 2전략이 동시에 기여할 수 있게 함
- 작업
  - 동일 방향 기존 포지션 스킵 로직 제거
  - delta order 기반 합산만 유지
- 대상
  - [main.py](/C:/Users/tjddb/bybit-quant/main.py)

### 4단계. sizing 일원화
- 목표
  - 포트폴리오 배분, 축소, cap가 실제 주문 수량까지 이어지게 함
- 작업
  - `allocate()` 결과의 `size_pct`와 ATR sizing을 일관된 notional model로 통합
- 대상
  - [src/portfolio/manager.py](/C:/Users/tjddb/bybit-quant/src/portfolio/manager.py)
  - [src/risk/manager.py](/C:/Users/tjddb/bybit-quant/src/risk/manager.py)
  - [main.py](/C:/Users/tjddb/bybit-quant/main.py)

### 5단계. funding_arb 자본/리스크 보강
- 목표
  - BTC/ETH pair별 자본 배분과 pair별 리스크 상태를 정확히 분리
- 작업
  - `capital_pct` 총합 기준 수정
  - monitor 상태를 coin별로 분리
  - dead config를 정리하거나 enforcement 추가
- 대상
  - [main.py](/C:/Users/tjddb/bybit-quant/main.py)
  - [src/risk/arb_monitor.py](/C:/Users/tjddb/bybit-quant/src/risk/arb_monitor.py)
  - [strategies/funding_arb/config.yaml](/C:/Users/tjddb/bybit-quant/strategies/funding_arb/config.yaml)

## 수정 후 최소 검증 항목
- ETH 전략이 `data/raw/bybit/ETHUSDTUSDT/funding_rate.parquet`를 읽는지 테스트
- BTC 모멘텀 + BTC 평균회귀 동시 신호 시 가상 포지션 2개가 유지되고 실제 delta order만 1회 생성되는지 테스트
- BTC funding arb short 보유 중 BTC v1 long 신호가 arb short를 상쇄하지 않도록 테스트
- `portfolio_scale`, `strategy_scale`, `max_symbol_exposure` 축소가 실제 주문 수량에 반영되는지 테스트
- funding arb에서 BTC/ETH 각각의 negative funding streak가 독립 집계되는지 테스트

## 최종 메모
현재 구조에서 가장 위험한 문제는 단순 성능 저하가 아니라:
- ETH 전략 입력 오염
- 동일 심볼 포지션 충돌
- 포트폴리오 배분 무효화

이 세 가지는 실거래 동작을 직접 바꾸므로 우선적으로 패치해야 한다.
