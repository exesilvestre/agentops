"""Regression check: detect metric drops vs a rolling baseline."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List, Optional

from agentops.agent.config import RegressionCheckConfig
from agentops.agent.findings import Category, Finding, Severity
from agentops.agent.sources.results_history import ResultsHistory, RunSummary
from agentops.core.results import RegressionInsight, RunResult
from agentops.pipeline.regression_insight import build_regression_insight, resolve_report_url


def _load_run_result(summary: RunSummary) -> Optional[RunResult]:
    """Best-effort reload of the full stored result behind a history entry.

    ``RunSummary`` is a thin projection with no commit/config fields; the
    causal explanation needs the full ``RunResult``. Only local runs have a
    real file to reload from (cloud-sourced summaries use a synthetic
    ``raw_path``); any read/parse failure is treated the same as "unknown"
    rather than raised, since this attribution is always best-effort.
    """
    if summary.source != "local":
        return None
    try:
        data = json.loads(summary.raw_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return RunResult.model_validate(data)
    except ValueError:
        return None


def _attribution_unavailable_reason(
    latest: RunSummary,
    previous_run: RunSummary,
    latest_result: Optional[RunResult],
    previous_result: Optional[RunResult],
) -> str:
    """Why ``build_regression_insight`` was skipped or returned ``None``.

    Surfaced instead of silently falling back to the generic
    recommendation, since a cloud-fetched run (Doctor's fallback to
    Foundry evaluation runs when local history is too short - see
    ``results_history._merge_runs``) has no local ``results.json`` to
    reload and therefore no recorded commit, unlike every run produced by
    ``agentops eval run`` (including ``execution: cloud``/``azd``), which
    always attempts commit capture.
    """
    non_local = [
        run
        for run, result in ((latest, latest_result), (previous_run, previous_result))
        if result is None and run.source != "local"
    ]
    if non_local:
        return "attribution unavailable: run has no commit (fetched from cloud)"
    if latest_result is None or previous_result is None:
        return "attribution unavailable: the run's results.json could not be read"
    if latest_result.commit is None or previous_result.commit is None:
        return "attribution unavailable: run has no recorded commit"
    return (
        "attribution unavailable: the regressed metric's value is missing "
        "on one of the compared runs"
    )


def run_regression_check(
    history: ResultsHistory,
    config: RegressionCheckConfig,
    *,
    workspace: Optional[Path] = None,
) -> List[Finding]:
    runs = history.runs
    if len(runs) < config.min_runs:
        return []

    latest = runs[-1]
    # Only compare against runs that share the same dataset, evaluators,
    # and agent identity - but deliberately version/deployment-*blind*
    # (`lineage_key`, not the finer `methodology_fingerprint` opex.py's
    # flaky-metric check needs, which does want version included since
    # mixing versions there would inflate variance spuriously). A version
    # bump is exactly the kind of change this check - and its causal
    # attribution - exists to catch, not an incompatible methodology to
    # exclude: keying this on the fingerprint instead would make the
    # check structurally unable to fire on the first run(s) after a
    # version bump (there being no baseline yet sharing that new,
    # not-yet-established fingerprint), making the feature's own
    # canonical "run v3 -> v4" scenario unreachable.
    lineage_key = latest.lineage_key
    if lineage_key is None:
        baseline_runs = runs[:-1]
    else:
        baseline_runs = [r for r in runs[:-1] if r.lineage_key == lineage_key]
    if len(baseline_runs) + 1 < config.min_runs:
        return []
    if not baseline_runs:
        return []

    # The immediately preceding comparable run, for causal attribution -
    # distinct from the rolling drop% mean below, which uses every
    # comparable run, not just the last one.
    previous_run = baseline_runs[-1]
    latest_result = _load_run_result(latest)
    previous_result = _load_run_result(previous_run)

    findings: List[Finding] = []
    for metric in config.metrics:
        baseline_values = [
            r.metrics[metric] for r in baseline_runs if metric in r.metrics
        ]
        if not baseline_values:
            continue
        if metric not in latest.metrics:
            continue

        baseline = mean(baseline_values)
        current = latest.metrics[metric]
        if baseline <= 0:
            continue

        drop = (baseline - current) / baseline
        if drop < config.threshold_drop:
            continue

        severity = (
            Severity.CRITICAL
            if drop >= max(config.threshold_drop * 2, 0.20)
            else Severity.WARNING
        )

        recommendation = (
            "Compare the latest run against the baseline runs in "
            "`.agentops/results/` or the Foundry Evaluations page, "
            "inspect prompt/model/dataset changes, and re-run the "
            "evaluation after the fix."
        )
        evidence = {
            "metric": metric,
            "current": current,
            "baseline_avg": baseline,
            "drop_ratio": drop,
            "baseline_runs": len(baseline_values),
            "latest_run_id": latest.run_id,
        }

        insight: Optional[RegressionInsight] = None
        if latest_result is not None and previous_result is not None:
            insight = build_regression_insight(
                previous_result,
                latest_result,
                metrics=[metric],
                workspace=workspace,
                # Both are already-completed historical runs, so a sidecar
                # `cloud_evaluation.json` next to either (from
                # `execution: cloud` or a finished `publish: true`) is
                # complete by now - unlike the in-flight `--baseline`
                # comparison in `pipeline.comparison`.
                from_report_url=resolve_report_url(
                    previous_result, results_path=previous_run.raw_path
                ),
                to_report_url=resolve_report_url(latest_result, results_path=latest.raw_path),
            )
        if insight is not None:
            evidence["insight"] = insight.model_dump(mode="json")
            recommendation = insight.explanation
            if insight.suggested_action:
                recommendation = f"{recommendation} {insight.suggested_action}"
        else:
            reason = _attribution_unavailable_reason(
                latest, previous_run, latest_result, previous_result
            )
            evidence["attribution_unavailable"] = reason
            recommendation = f"{recommendation} ({reason})"

        findings.append(
            Finding(
                id=f"regression.{metric}",
                severity=severity,
                category=Category.QUALITY,
                title=f"Regression detected on `{metric}`",
                summary=(
                    f"`{metric}` dropped {drop * 100:.1f}% in run "
                    f"`{latest.run_id}` (current={current:.4f}, "
                    f"baseline={baseline:.4f} over {len(baseline_values)} runs)."
                ),
                recommendation=recommendation,
                source="results_history",
                evidence=evidence,
            )
        )
    return findings
