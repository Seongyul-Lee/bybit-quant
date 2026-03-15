"""Tests for the Phase 1 validation gate and artifact persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.validation.gate import (
    GATE_VERSION,
    decide_model_replacement,
    evaluate_deployment_gate,
    save_strategy_validation_artifact,
    save_validation_run_summary,
)


class TestEvaluateDeploymentGate:
    def test_strict_oos_failure_blocks_pass(self) -> None:
        conservative = {
            "pv_pf": 1.8,
            "pv_return": 2.4,
            "pv_trades": 32,
            "pf_drop": 18.0,
        }
        strict_oos = {
            "pf": 0.72,
            "total_return": -1.5,
            "trades": 14,
        }

        result = evaluate_deployment_gate(conservative, strict_oos)

        assert result["gate_version"] == GATE_VERSION
        assert result["passed"] is False
        assert any("strict_oos.pf" in reason for reason in result["failure_reasons"])
        assert any("strict_oos.total_return" in reason for reason in result["failure_reasons"])

    def test_no_trade_strict_oos_is_explicit_failure(self) -> None:
        conservative = {
            "pv_pf": 1.9,
            "pv_return": 1.1,
            "pv_trades": 30,
            "pf_drop": 12.0,
        }
        strict_oos = {
            "pf": 0.0,
            "total_return": 0.0,
            "trades": 0,
        }

        result = evaluate_deployment_gate(conservative, strict_oos)

        assert result["passed"] is False
        assert any("strict_oos.trades" in reason for reason in result["failure_reasons"])


class TestArtifactPersistence:
    def test_strategy_artifact_path_and_schema_are_stable(self, tmp_path) -> None:
        generated_at = datetime(2026, 3, 15, tzinfo=timezone.utc)
        payload = {
            "strategy": "btc_1h_momentum",
            "generated_at": generated_at.isoformat(),
            "passed": False,
            "failure_reasons": ["strict_oos.pf failed"],
        }

        saved = save_strategy_validation_artifact(
            payload,
            strategy_name="btc_1h_momentum",
            output_dir=tmp_path,
            timestamp=generated_at,
        )

        artifact_path = tmp_path / "strategies" / "btc_1h_momentum" / "20260315T000000Z.json"
        assert artifact_path.exists()
        assert saved["artifact"]["type"] == "strategy_validation"
        assert saved["artifact"]["path"] == artifact_path.as_posix()

        persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert persisted["strategy"] == "btc_1h_momentum"
        assert persisted["artifact"]["path"] == artifact_path.as_posix()

    def test_run_summary_artifact_uses_run_directory(self, tmp_path) -> None:
        generated_at = datetime(2026, 3, 15, tzinfo=timezone.utc)
        payload = {
            "generated_at": generated_at.isoformat(),
            "strategy_count": 2,
            "passed_count": 1,
            "failed_count": 1,
        }

        saved = save_validation_run_summary(
            payload,
            output_dir=tmp_path,
            timestamp=generated_at,
        )

        artifact_path = tmp_path / "runs" / "20260315T000000Z.json"
        assert artifact_path.exists()
        assert saved["artifact"]["type"] == "validation_run_summary"
        assert saved["artifact"]["path"] == artifact_path.as_posix()


class TestReplacementDecision:
    def test_gate_failure_blocks_model_replacement(self) -> None:
        old_oos = {"pv_pf": 2.0}
        new_oos = {
            "pv_pf": 1.95,
            "passed": False,
            "failure_reasons": ["strict_oos.pf failed: expected >= 1.00, actual=0.72"],
        }

        should_replace, reason = decide_model_replacement(old_oos, new_oos, min_pf_ratio=0.9)

        assert should_replace is False
        assert "validation gate failed" in reason
        assert "strict_oos.pf" in reason

    def test_pf_ratio_still_applies_after_gate_pass(self) -> None:
        old_oos = {"pv_pf": 2.0}
        new_oos = {
            "pv_pf": 1.4,
            "passed": True,
            "failure_reasons": [],
        }

        should_replace, reason = decide_model_replacement(old_oos, new_oos, min_pf_ratio=0.9)

        assert should_replace is False
        assert "PV PF degraded too much" in reason
