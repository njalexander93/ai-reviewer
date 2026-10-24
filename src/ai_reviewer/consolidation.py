"""Consolidate validated review coverage into a source-preserving publication artifact.

The canonical loader owns manifest replay. This module batches finding groups and
summary reductions, calls the selected consolidator, and rejects incomplete partitions.
"""

from pathlib import Path
from typing import Any

from ai_reviewer.contracts import (
    Consolidation,
    FinalConsolidation,
    Group,
    ReviewContext,
    reconcile,
    source_findings,
)
from ai_reviewer.limits import (
    CONSOLIDATION_BYTES,
    MAX_CONSOLIDATION_CALLS,
    MAX_FINDINGS,
    MAX_SUMMARY_CALLS,
    SUMMARY_PAGE_SIZE,
)
from ai_reviewer.models import CONSOLIDATORS, call_model
from ai_reviewer.serialization import encode
from ai_reviewer.work import load_reviews


def consolidate_chunks(context: ReviewContext, directory: Path) -> None:
    """Group original findings and reduce assessments through bounded consolidator calls.

    Args:
        context: Snapshot used to validate coverage before any consolidation call.
        directory: Run root with complete review artifacts and no consolidation directories.

    Raises:
        ValueError: Coverage fails, prior consolidation artifacts exist, call budgets
            are exhausted, or consolidator drops, duplicates or invents source IDs.
        Exception: Provider and filesystem failures propagate with available usage retained.

    Writes per-call artifacts, consolidate.json and coverage.txt. Cross-batch duplicate
    findings remain separate. Original finding text and severity are never replaced;
    summary semantic fidelity depends on the model, not the partition checks.
    """
    # Reject retries that would overwrite earlier cost evidence; replay coverage first.
    if any((directory / name).exists() for name in ("consolidation", "consolidation-summaries")):
        raise ValueError("Consolidation requires fresh artifacts; earlier usage must be retained")
    covered = load_reviews(context, directory)
    consolidator = CONSOLIDATORS[covered.providers[0]]
    reports = covered.reviews
    sources = source_findings(reports)
    items = list(sources.items())
    groups: list[Group] = []
    summaries = [r.summary for r in reports]
    # Bound each consolidation call by bytes and source count, retaining global IDs.
    batches: list[dict[str, Any]] = []
    batch: dict[str, Any] = {}
    for key, finding in items:
        trial = {**batch, key: finding.model_dump()}
        if (
            len(trial) > MAX_FINDINGS
            or len(encode({"sources": trial}).encode()) > CONSOLIDATION_BYTES
        ):
            batches.append(batch)
            batch = {}
        batch[key] = finding.model_dump()
    if batch or not batches:
        batches.append(batch)
    if len(batches) > MAX_CONSOLIDATION_CALLS:
        raise ValueError("Consolidation work budget exceeded")
    # Small reviews carry original assessments in their single grouping call.
    combined = encode({"sources": batches[0], "reviews": summaries})
    include_summaries = len(batches) == 1 and len(combined.encode()) <= CONSOLIDATION_BYTES
    if include_summaries:
        summaries = []
    # Validate an exact source partition after each paid grouping call.
    for index, sources_batch in enumerate(batches):
        target = directory / "consolidation" / str(index)
        payload = combined if include_summaries else encode({"sources": sources_batch})
        call_model(consolidator, payload, target)
        result = Consolidation.model_validate_json((target / f"{consolidator}.json").read_text())
        flattened = [key for group in result.groups for key in group.source_ids]
        if sorted(flattened) != sorted(sources_batch):
            raise ValueError("Consolidation lost or duplicated original findings")
        groups.extend(result.groups)
        summaries.append(result.summary)
    # Summarize bounded pages of original assessments and grouping results. No finding
    # text is rewritten; explicit disagreements must survive every summary round.
    summary_calls = 0
    while len(summaries) > 1:
        reduced = []
        for start in range(0, len(summaries), SUMMARY_PAGE_SIZE):
            target = directory / "consolidation-summaries" / str(summary_calls)
            summary_calls += 1
            if summary_calls > MAX_SUMMARY_CALLS:
                raise ValueError("Summary work budget exceeded")
            call_model(
                consolidator,
                encode({"sources": {}, "reviews": summaries[start : start + SUMMARY_PAGE_SIZE]}),
                target,
            )
            result = Consolidation.model_validate_json(
                (target / f"{consolidator}.json").read_text()
            )
            if result.groups:
                raise ValueError("Summary pass invented findings")
            reduced.append(result.summary)
        summaries = reduced
    # Reconcile globally before writing the one publishable consolidation result.
    result = FinalConsolidation(summary=summaries[0], groups=groups)
    reconcile(reports, result)
    (directory / "consolidate.json").write_text(result.model_dump_json())
    (directory / "coverage.txt").write_text(covered.summary)
