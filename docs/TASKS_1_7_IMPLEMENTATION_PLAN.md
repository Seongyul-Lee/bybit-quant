# Tasks 1-7 Implementation Plan

Date: 2026-03-15

## Goal

Turn the current pre-mainnet research stack into a stricter deployment candidate by completing the seven priority tasks:

1. Put `strict OOS` directly into the deployment gate.
2. Reduce the deployment set to one strategy first.
3. Make fold and threshold selection deterministic.
4. Align backtest sizing and live sizing.
5. Replace weighted-return portfolio backtest with shared-capital simulation.
6. Make funding arbitrage rules unambiguous in code and docs.
7. Persist validation artifacts automatically after retrain/validation.

This plan is designed so the work can be executed across several user requests without losing continuity.

## Current Baseline

- Latest saved OOS result is [docs/oos_validation_2026-03-15.json](/C:/Users/tjddb/bybit-quant/docs/oos_validation_2026-03-15.json).
- `passed=true` is currently based on post-validation rules, not strict OOS.
- `strict_oos` is negative for all three ML strategies in the latest saved output.
- Mainnet is not deployed yet, so this plan assumes pre-deployment refactor and validation work.

## Working Rules

- No strategy is treated as deployable unless it passes the new strict gate.
- Backtest assumptions and live execution assumptions must come from the same source of truth.
- All validation outputs must be reproducible from saved artifacts.
- Strategy count is reduced before portfolio complexity is increased.

## Phase 1

### Scope

Task 1 and Task 7.

### Objective

Redefine validation so that `strict OOS` can block deployment, then persist every result automatically.

### Files to change

- [oos_validation.py](/C:/Users/tjddb/bybit-quant/oos_validation.py)
- [retrain.py](/C:/Users/tjddb/bybit-quant/retrain.py)
- [README.md](/C:/Users/tjddb/bybit-quant/README.md)
- New folder under `reports/` or `artifacts/` for validation snapshots
- New tests under [tests](/C:/Users/tjddb/bybit-quant/tests)

### Planned changes

- Change validation output schema so it records:
  - gate version
  - exact criteria used
  - post-validation result
  - strict OOS result
  - final deployment decision
  - failure reasons
- Make `strict OOS` part of the final pass/fail decision.
- Save JSON output automatically on every validation run.
- Save one artifact per strategy and one combined summary artifact for `--all`.
- Update `retrain.py` so model replacement is blocked when the new gate fails.
- Add tests covering:
  - strict OOS failure blocks pass
  - no-trade periods are handled explicitly
  - saved artifact path and schema are stable

### Acceptance criteria

- Running validation creates a timestamped JSON artifact without manual copy/paste.
- A strategy with negative strict OOS cannot produce `passed=true`.
- Retrain logs show the new gate decision and the failure reason.

## Phase 2

### Scope

Task 2 and Task 3.

### Objective

Freeze the deployment universe to one candidate strategy and eliminate discretionary fold picking.

### Files to change

- [config/portfolio.yaml](/C:/Users/tjddb/bybit-quant/config/portfolio.yaml)
- [strategies/_common/trainer.py](/C:/Users/tjddb/bybit-quant/strategies/_common/trainer.py)
- Strategy configs under [strategies](/C:/Users/tjddb/bybit-quant/strategies)
- [README.md](/C:/Users/tjddb/bybit-quant/README.md)
- Optional new deployment config file under `config/`

### Planned changes

- Mark only one ML strategy as deployment candidate.
- Move the other ML strategies to research-only status.
- Replace manual ensemble fold selection with a deterministic rule. Example options:
  - latest eligible fold only
  - latest N eligible folds
  - median-of-recent eligible folds
- Record the selection rule and chosen folds inside saved artifacts.
- Make threshold selection rule explicit and reproducible.
- Update docs so README no longer reads like all strategies are equally deployment-ready.

### Acceptance criteria

- `portfolio.yaml` exposes only one ML strategy as deployable.
- Fold selection no longer depends on manually curated fold lists inside strategy config.
- Validation artifacts show exactly why the chosen model/folds were selected.

## Phase 3

### Scope

Task 4.

### Objective

Make backtest and live execution use the same sizing and execution assumptions.

### Files to change

- [backtest.py](/C:/Users/tjddb/bybit-quant/backtest.py)
- [main.py](/C:/Users/tjddb/bybit-quant/main.py)
- [src/risk/manager.py](/C:/Users/tjddb/bybit-quant/src/risk/manager.py)
- [src/portfolio/manager.py](/C:/Users/tjddb/bybit-quant/src/portfolio/manager.py)
- Optional shared simulation/execution module under `src/`

### Planned changes

- Define one reusable execution model for:
  - fee
  - slippage
  - max holding period
  - stop loss / take profit
  - ATR-based sizing cap
  - portfolio allocation cap
- Refactor backtest to call the same sizing logic used by live code, or refactor both to call a shared module.
- Make the backtest output show the exact assumptions used.
- Add tests that compare position sizing outputs between live and backtest code paths.

### Acceptance criteria

- The same input state yields the same position size and stop/take-profit parameters in both backtest and live code.
- Assumption drift between backtest and live paths is removed or explicitly documented in one place.

## Phase 4

### Scope

Task 5 and Task 6.

### Objective

Replace approximate portfolio validation with shared-capital simulation and make funding arbitrage behavior explicit.

### Files to change

- [portfolio_backtest.py](/C:/Users/tjddb/bybit-quant/portfolio_backtest.py)
- [strategies/funding_arb/strategy.py](/C:/Users/tjddb/bybit-quant/strategies/funding_arb/strategy.py)
- [backtest_funding_arb.py](/C:/Users/tjddb/bybit-quant/backtest_funding_arb.py)
- [config/portfolio.yaml](/C:/Users/tjddb/bybit-quant/config/portfolio.yaml)
- Funding arb docs under [docs](/C:/Users/tjddb/bybit-quant/docs)

### Planned changes

- Replace weighted return summation with a shared-capital state machine:
  - cash
  - reserved margin
  - concurrent exposure
  - strategy-level position state
  - portfolio-level risk limits
- Decide and encode one funding arbitrage mode:
  - always-on hedge
  - rule-based entry/exit
  - model-driven entry/exit
- Remove dead branches where research artifacts imply a different live strategy than the code actually executes.
- Make the portfolio backtest include the funding sleeve if it is part of the real intended deployment.

### Acceptance criteria

- Portfolio backtest reflects shared capital and concurrent positions rather than simple weighted returns.
- Funding arbitrage behavior is defined in one sentence and matches code, config, and docs.

## Phase 5

### Scope

Integration hardening after Phases 1-4.

### Objective

Re-run end-to-end validation with the new rules and document the resulting deployment status.

### Files to change

- [README.md](/C:/Users/tjddb/bybit-quant/README.md)
- [docs](/C:/Users/tjddb/bybit-quant/docs)
- Test files under [tests](/C:/Users/tjddb/bybit-quant/tests)

### Planned changes

- Re-run training and validation for the surviving strategy.
- Regenerate artifacts using the new schema.
- Update README tables so they reflect current gate definitions and current candidate status.
- Write one short deployment-readiness note that answers:
  - what passed
  - what failed
  - what remains blocked before mainnet

### Acceptance criteria

- Repo contains one current source of truth for strategy status.
- A new reader can tell which strategy, if any, is deployable without inferring from multiple conflicting files.

## Suggested Request Breakdown

If this work is split across multiple chats, use this order:

1. Request 1: implement Phase 1 only.
2. Request 2: implement Phase 2 only.
3. Request 3: implement Phase 3 only.
4. Request 4: implement Phase 4 only.
5. Request 5: run Phase 5 cleanup and documentation updates.

This ordering minimizes rework because the validation gate must be fixed before strategy selection, and strategy selection must be fixed before portfolio-level refactor.

## Definition of Done

The plan is complete only when all of the following are true:

- Validation pass/fail is driven by strict OOS-aware rules.
- One ML strategy is clearly designated as deployment candidate.
- Fold and threshold choice are deterministic and auditable.
- Backtest and live sizing share one logic path.
- Portfolio backtest models shared capital.
- Funding arbitrage logic is unambiguous.
- Validation artifacts are automatically saved and reproducible.
