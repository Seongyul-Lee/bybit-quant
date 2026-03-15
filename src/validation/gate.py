"""Validation gate and artifact helpers for OOS evaluation."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATE_VERSION = "strict_oos_v1"

DEFAULT_GATE_CRITERIA: dict[str, dict[str, float | int]] = {
    "post_validation": {
        "pf_min": 1.20,
        "return_pct_min_exclusive": 0.0,
        "trades_min": 25,
        "pf_drop_pct_max": 50.0,
    },
    "strict_oos": {
        "pf_min": 1.00,
        "return_pct_min_exclusive": 0.0,
        "trades_min": 10,
    },
}


def _format_reason(metric: str, expected: str, actual: float | int) -> str:
    return f"{metric} failed: expected {expected}, actual={actual}"


def evaluate_deployment_gate(
    conservative: dict[str, Any],
    strict_oos: dict[str, Any],
    criteria: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Evaluate the deployment gate using post-validation and strict OOS."""

    gate_criteria = deepcopy(criteria or DEFAULT_GATE_CRITERIA)
    failures: list[str] = []

    pv_rules = gate_criteria["post_validation"]
    pv_pf = float(conservative.get("pv_pf", 0) or 0)
    pv_return = float(conservative.get("pv_return", 0) or 0)
    pv_trades = int(conservative.get("pv_trades", 0) or 0)
    pf_drop = float(conservative.get("pf_drop", 100) or 100)

    if pv_pf < float(pv_rules["pf_min"]):
        failures.append(
            _format_reason("post_validation.pf", f">= {pv_rules['pf_min']:.2f}", pv_pf)
        )
    if pv_return <= float(pv_rules["return_pct_min_exclusive"]):
        failures.append(
            _format_reason(
                "post_validation.total_return",
                f"> {pv_rules['return_pct_min_exclusive']:.2f}",
                pv_return,
            )
        )
    if pv_trades < int(pv_rules["trades_min"]):
        failures.append(
            _format_reason(
                "post_validation.trades",
                f">= {int(pv_rules['trades_min'])}",
                pv_trades,
            )
        )
    if pf_drop > float(pv_rules["pf_drop_pct_max"]):
        failures.append(
            _format_reason(
                "post_validation.pf_drop_pct",
                f"<= {pv_rules['pf_drop_pct_max']:.2f}",
                pf_drop,
            )
        )

    strict_rules = gate_criteria["strict_oos"]
    strict_pf = float(strict_oos.get("pf", 0) or 0)
    strict_return = float(strict_oos.get("total_return", 0) or 0)
    strict_trades = int(strict_oos.get("trades", 0) or 0)

    if strict_pf < float(strict_rules["pf_min"]):
        failures.append(
            _format_reason("strict_oos.pf", f">= {strict_rules['pf_min']:.2f}", strict_pf)
        )
    if strict_return <= float(strict_rules["return_pct_min_exclusive"]):
        failures.append(
            _format_reason(
                "strict_oos.total_return",
                f"> {strict_rules['return_pct_min_exclusive']:.2f}",
                strict_return,
            )
        )
    if strict_trades < int(strict_rules["trades_min"]):
        failures.append(
            _format_reason(
                "strict_oos.trades",
                f">= {int(strict_rules['trades_min'])}",
                strict_trades,
            )
        )

    return {
        "gate_version": GATE_VERSION,
        "criteria": gate_criteria,
        "passed": not failures,
        "failure_reasons": failures,
    }


def decide_model_replacement(
    old_oos: dict[str, Any] | None,
    new_oos: dict[str, Any],
    min_pf_ratio: float,
) -> tuple[bool, str]:
    """Decide whether a retrained model may replace the current one."""

    if not new_oos.get("passed", False):
        reasons = new_oos.get("failure_reasons") or ["deployment gate failed"]
        return False, "validation gate failed: " + "; ".join(reasons)

    old_pv_pf = float((old_oos or {}).get("pv_pf", 0) or 0)
    new_pv_pf = float(new_oos.get("pv_pf", 0) or 0)

    if old_pv_pf > 0 and new_pv_pf < old_pv_pf * min_pf_ratio:
        threshold = old_pv_pf * min_pf_ratio
        return (
            False,
            f"PV PF degraded too much: {old_pv_pf:.2f} -> {new_pv_pf:.2f} "
            f"(minimum allowed {threshold:.2f})",
        )

    if old_pv_pf > 0:
        return True, f"replacement approved: PV PF {old_pv_pf:.2f} -> {new_pv_pf:.2f}"
    return True, f"replacement approved: PV PF N/A -> {new_pv_pf:.2f}"


def build_artifact_timestamp(now: datetime | None = None) -> str:
    """Build a stable UTC timestamp used in validation artifact paths."""

    timestamp = now or datetime.now(timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def _write_artifact(path: Path, payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    saved_payload = deepcopy(payload)
    saved_payload["artifact"] = {
        "type": artifact_type,
        "path": path.as_posix(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(saved_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return saved_payload


def save_strategy_validation_artifact(
    payload: dict[str, Any],
    strategy_name: str,
    output_dir: str | Path = "reports/validation",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Persist a per-strategy validation artifact."""

    stamp = build_artifact_timestamp(timestamp)
    path = Path(output_dir) / "strategies" / strategy_name / f"{stamp}.json"
    return _write_artifact(path, payload, "strategy_validation")


def save_validation_run_summary(
    payload: dict[str, Any],
    output_dir: str | Path = "reports/validation",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Persist a multi-strategy validation summary artifact."""

    stamp = build_artifact_timestamp(timestamp)
    path = Path(output_dir) / "runs" / f"{stamp}.json"
    return _write_artifact(path, payload, "validation_run_summary")
