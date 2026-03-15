# Mainnet 단계적 배포 계획

## 개요

funding_arb → v1 추가 → 전체 자본 순서로 2개월 단위 단계적 배포.
각 단계에서 검증 기준을 충족해야 다음 단계로 진행.

---

## Phase 1: arb 단독 (1~2개월차)

### 자본 배분

```
총 투입: $500 (전체 $1,400 중)
  BTC 차익: $250 (현물 $125 + 선물 숏 $125, 2x 레버리지)
  ETH 차익: $250 (현물 $125 + 선물 숏 $125, 2x 레버리지)
  나머지 $900: Bybit 계정 내 USDT 보관 (마진 여유)
```

### 기대 수익

```
백테스트 기준: 연 14.23% (2x)
실전 보수적 추정 (× 0.6): 연 ~8.5%
2개월 기대 수익: $500 × 8.5% × (2/12) = ~$7
2개월 최악 손실: $500 × -3% = -$15
```

### 설정

```yaml
# config/portfolio.yaml
portfolio:
  active_strategies: []  # v1 비활성화

  funding_arb:
    enabled: true
    config_path: strategies/funding_arb/config.yaml
    capital_pct: 0.36  # $500 / $1,400 ≈ 36%
```

```yaml
# strategies/funding_arb/config.yaml
params:
  leverage: 2
```

### 실행 전 체크리스트

```
[ ] mainnet API 키 확인 (.env에 BYBIT_API_KEY, BYBIT_SECRET)
[ ] Bybit mainnet 계정에 $1,400 이상 USDT 확인
[ ] config 복원:
    [ ] confidence_threshold: 0.44 / 0.44 / 0.50
    [ ] volatility_threshold: 0.05
[ ] portfolio.yaml: active_strategies를 빈 리스트로 설정 (v1 비활성화)
[ ] portfolio.yaml: funding_arb.enabled: true
[ ] portfolio.yaml: funding_arb.capital_pct: 0.36
[ ] Hedge Mode 전환은 arb 단독이라 불필요 (One-Way Mode로 충분)
    → 단, Phase 2에서 v1 추가 시 전환 필요하므로 미리 해도 무방
[ ] VPS에서 git pull origin preview
[ ] testnet 상태 파일과 mainnet 상태 파일 분리 확인
    (config/current_state_testnet.json vs config/current_state.json)
```

### 실행 명령

```bash
# VPS에서
cd ~/bybit-quant
source .venv/bin/activate
git pull origin preview

# mainnet 실행 (--testnet 플래그 없음)
tmux new -s quant-live
python main.py --mode live

# Ctrl+B → D로 detach
```

### 일일 모니터링

```
매일 확인:
  - 텔레그램 알림 정상 수신
  - 비대칭 포지션 알림 없음
  - 펀딩비 결제 알림 (09:00, 17:00, 01:00 KST)

매주 확인:
  - 누적 펀딩비 수취 금액
  - 델타 이력 (< 5% 유지)
  - 베이시스 이력 (mainnet은 ±0.1% 수준 정상)
  - USDT 잔고 변화

매월 기록:
  - 총 수익/손실 (USDT)
  - 실전 수익률 vs 백테스트 비율
  - 인프라 장애 횟수
  - 펀딩비 양수/음수 비율
```

### Phase 2 진입 기준 (2개월 후)

```
필수 (3/3 충족):
  ✅ 2개월 누적 수익 > 0 (수수료 차감 후)
  ✅ 무크래시 운영 또는 자동 복구 성공
  ✅ 델타 이탈 사고 0건

권장 (2/3 충족):
  ✅ 실전 수익률 > 백테스트 × 0.5
  ✅ 펀딩비 양수 비율 > 70%
  ✅ 최대 일간 손실 < -1%
```

### Phase 1 중단 조건

```
즉시 중단 + 전체 청산:
  - 누적 손실 > -5% ($25)
  - 비대칭 포지션 24시간 이상 미해결
  - Bybit 이상 징후 (출금 제한, API 장기 장애)

축소 (포지션 50% 감소):
  - 연속 음수 펀딩비 1주일 (21회)
  - 월간 손실 > -2%

관찰 (유지하되 주시):
  - 베이시스 ±1% 이상 지속 1일
  - 펀딩비 양수 비율 < 60% (월간)
```

---

## Phase 2: arb + v1 추가 (3~4개월차)

### 자본 배분

```
총 투입: $1,000 (추가 $500)
  funding_arb: $400 (40%)
    BTC: $200 (현물 $100 + 선물 $100)
    ETH: $200 (현물 $100 + 선물 $100)
  v1 3롱: $450 (45%)
    btc_1h_momentum: $150
    eth_1h_momentum: $150
    btc_1h_mean_reversion: $150
  현금 버퍼: $150 (15%)
```

### 설정 변경

```yaml
# config/portfolio.yaml
portfolio:
  active_strategies:
    - btc_1h_momentum
    - eth_1h_momentum
    - btc_1h_mean_reversion

  allocation:
    position_pct_per_strategy: 0.15  # $1,000 × 15% = $150/전략

  funding_arb:
    enabled: true
    capital_pct: 0.40
```

### 추가 설정

```
[ ] Hedge Mode 전환 (v1 + arb 동시 운용 필수)
    → 전환 전 모든 포지션 0 확인
    → arb 포지션 청산 → Hedge Mode 전환 → arb 재진입
[ ] v1 모델 재학습 (배포 전 최신 데이터로)
    → python retrain.py --all
[ ] confidence_threshold 원래 값 확인 (0.44/0.44/0.50)
```

### Phase 3 진입 기준 (4개월 후)

```
필수:
  ✅ v1 실전 Profit Factor > 0.8
  ✅ 포트폴리오 합산 MDD < -10%
  ✅ arb 여전히 정상 (누적 양수)
  ✅ Hedge Mode 안정 운영 (v1/arb 포지션 충돌 0건)

권장:
  ✅ v1 실전 수익률 > 0 (2개월간)
  ✅ 합산 수익률 > arb 단독 (v1이 알파 제공)
```

### Phase 2 중단 조건

```
v1 중단 (arb만 유지):
  - v1 실전 PF < 0.5 (2개월간)
  - v1 MDD > -8%
  - v1이 arb 수익을 완전 상쇄

전체 중단:
  - 포트폴리오 MDD > -15%
  - Hedge Mode 오작동 (v1/arb 포지션 상쇄 발생)
  - 거래소 리스크 징후
```

---

## Phase 3: 전체 자본 (5~6개월차)

### 자본 배분

```
총 투입: $1,400 (전체)
  funding_arb: $560 (40%)
    BTC: $280
    ETH: $280
  v1 3롱: $630 (45%)
    btc_1h_momentum: $210
    eth_1h_momentum: $210
    btc_1h_mean_reversion: $210
  현금 버퍼: $210 (15%)
```

### 이 시점에서 확보된 것

```
6개월 트랙 레코드:
  - arb: 6개월 실전 데이터
  - v1: 4개월 실전 데이터
  - 실전/백테스트 비율 확정
  - 인프라 안정성 검증 완료

판단 가능한 것:
  - 추가 자본 투입 근거 (자본 증액 or 현상 유지 or 축소)
  - 전략 추가/제거 판단 (v1 전략별 성과 기반)
  - 레버리지 조정 판단 (arb 2x → 3x 검토)
```

---

## 공통 운영 규칙

### 텔레그램 알림 대응 매뉴얼

```
🚨 긴급 (즉시 확인):
  "포트폴리오 MDD 한도 도달"     → VPS 로그 확인, 원인 파악
  "마진 긴급"                    → Bybit 앱에서 잔고 확인
  "비대칭 포지션"                → 수동으로 현물 매도 또는 선물 숏 진입
  "Circuit Breaker 발동"        → 시장 상황 확인 후 수동 리셋 판단

⚠️ 경고 (1시간 내 확인):
  "베이시스 긴급"                → mainnet에서는 드묾, 지속 시 조사
  "델타 초과"                    → 자동 복구 여부 확인, 미복구 시 수동 조치
  "연속 음수 펀딩비"             → 시장 센티먼트 확인, 극단적이면 축소 검토

ℹ️ 정보 (일일 확인):
  "펀딩비 결제 확인"             → 금액 기록
  "주문 실행"                    → 정상 확인
  "실거래 시작"                  → 재시작 이력 확인
```

### VPS 유지보수

```
매주:
  - tmux 세션 확인 (프로세스 살아있는지)
  - 디스크 용량 확인: df -h
  - 로그 크기 확인: du -sh logs/

매월:
  - Oracle Cloud 크레딧 확인
  - VPS OS 업데이트: sudo apt update && sudo apt upgrade
  - Python 패키지 업데이트 검토 (ccxt 버전 등)
  - git pull로 코드 최신화 (패치 적용 시)

분기:
  - 모델 재학습: python retrain.py --all
  - OOS 검증 재실행
  - 전략 성과 리뷰 + 존폐 판단
```

### 긴급 복구 절차

```
봇 크래시 시:
  1. ssh 접속
  2. tmux attach -t quant-live
  3. 로그 확인 (에러 원인)
  4. python main.py --mode live  (재시작)
  5. 상태 파일이 자동 복원됨

비대칭 포지션 발생 시:
  1. 봇 중단 (Ctrl+C)
  2. 현물 잔고 확인 + 선물 포지션 확인
  3. 부족한 쪽 수동 실행:
     - 현물만 있으면: 현물 매도
     - 선물만 있으면: 선물 청산
  4. 상태 파일 리셋
  5. 봇 재시작

거래소 장애 시:
  1. Bybit 공식 채널 확인 (status.bybit.com)
  2. 봇 중단 (불필요한 재시도 방지)
  3. 장애 복구 후 포지션 확인
  4. 정상이면 재시작, 비정상이면 수동 정리 후 재시작
```

---

## 타임라인 요약

```
2026-03-15  testnet 검증 완료 (펀딩비 수취 확인)
2026-03-16  Phase 1 시작: arb 단독 $500 mainnet 배포
2026-05-16  Phase 1 검증 완료 (2개월)
2026-05-17  Phase 2 시작: v1 추가, $1,000 배포
2026-07-17  Phase 2 검증 완료 (2개월)
2026-07-18  Phase 3 시작: 전체 $1,400 배포
2026-09-15  6개월 트랙 레코드 확보 → 자본 증액 판단
```

---

## 성공 기준 (6개월 후)

```
최소 목표:
  - 누적 수익 > 0 (원금 보전)
  - 시스템 가동률 > 95%
  - 실전/백테스트 비율 > 0.4

기본 목표:
  - 누적 수익률 > +5% ($70)
  - 실전/백테스트 비율 > 0.6
  - MDD < -5%

낙관 목표:
  - 누적 수익률 > +10% ($140)
  - 실전/백테스트 비율 > 0.7
  - 자본 증액 근거 확보
```
