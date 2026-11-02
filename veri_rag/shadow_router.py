"""
5% Shadow Traffic Router, Trajectory Diff Comparator & Paired Bootstrap
Statistical Significance Engine (1,000 Resampling Iterations).

Mirrors a configurable fraction (e.g., 5% production shadow traffic or 100% in
benchmark mode) of live RAG requests to a candidate pipeline asynchronously on a
background worker pool without blocking primary response latency.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import random
import time
from typing import Optional

from veri_rag.adversarial_harness import (
    AdversarialAbstentionHarness,
    GenerationOutput,
    PerturbedTrial,
    RAGPipelineProtocol,
)
from veri_rag.hybrid_retriever import tokenize


@dataclass(frozen=True)
class ShadowComparisonRecord:
    """Paired observation comparing Primary vs. Candidate pipeline on a single shadowed query."""

    trial_id: str
    query: str
    primary_answer: str
    candidate_answer: str
    primary_abstained: bool
    candidate_abstained: bool
    primary_fchr_violation: bool
    candidate_fchr_violation: bool
    primary_citation_precision: float
    candidate_citation_precision: float
    primary_latency_ms: float
    candidate_latency_ms: float
    primary_caller_overhead_ms: float
    primary_cost_usd: float
    candidate_cost_usd: float
    answer_jaccard_similarity: float
    trajectory_diffs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BootstrapMetricCI:
    """Paired bootstrap statistical comparison result for a single metric."""

    metric_name: str
    mean_primary: float
    mean_candidate: float
    mean_delta: float
    ci_lower: float
    ci_upper: float
    half_width: float
    p_value: float
    is_statistically_significant: bool
    formatted_summary: str


@dataclass(frozen=True)
class ShadowRouterReport:
    """Comprehensive report from a Shadow Traffic Routing run."""

    total_queries_received: int
    shadowed_queries_count: int
    configured_sample_rate: float
    actual_sample_rate: float
    mean_primary_caller_overhead_ms: float
    agreement_rate: float
    fchr_bootstrap: BootstrapMetricCI
    citation_precision_bootstrap: BootstrapMetricCI
    latency_ms_bootstrap: BootstrapMetricCI
    cost_usd_bootstrap: BootstrapMetricCI
    records: list[ShadowComparisonRecord]


class ShadowTrafficRouter:
    """
    Asynchronous Shadow Traffic Router & Paired Bootstrap Comparator.
    """

    def __init__(
        self,
        primary_pipeline: RAGPipelineProtocol,
        candidate_pipeline: RAGPipelineProtocol,
        sample_rate: float = 0.05,
        random_seed: int = 42,
        bootstrap_iterations: int = 1000,
        max_workers: int = 4,
    ) -> None:
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        self.primary_pipeline = primary_pipeline
        self.candidate_pipeline = candidate_pipeline
        self.sample_rate = sample_rate
        self.rng = random.Random(random_seed)
        self.bootstrap_iterations = bootstrap_iterations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_futures: list[Future[ShadowComparisonRecord]] = []
        self._completed_records: list[ShadowComparisonRecord] = []
        self.total_requests = 0

    @staticmethod
    def _compute_answer_jaccard(ans_a: str, ans_b: str) -> float:
        set_a = set(tokenize(ans_a, remove_stopwords=True))
        set_b = set(tokenize(ans_b, remove_stopwords=True))
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 1.0
        return round(len(set_a & set_b) / len(union), 4)

    @staticmethod
    def _compute_trajectory_diffs(
        primary_out: GenerationOutput, candidate_out: GenerationOutput
    ) -> list[str]:
        diffs: list[str] = []
        if primary_out.abstained != candidate_out.abstained:
            diffs.append(
                f"ABSTENTION_FLIP: primary={primary_out.abstained} -> candidate={candidate_out.abstained} "
                f"(reason={candidate_out.abstention_reason})"
            )
        if primary_out.conflict_detected != candidate_out.conflict_detected:
            diffs.append(
                f"CONFLICT_RESOLUTION: candidate dropped superseded docs {candidate_out.superseded_docs_dropped}"
            )
        if primary_out.cited_doc_ids != candidate_out.cited_doc_ids:
            diffs.append(
                f"CITATION_SHIFT: primary={primary_out.cited_doc_ids} -> candidate={candidate_out.cited_doc_ids}"
            )
        return diffs

    def _run_shadow_comparison(
        self,
        trial: PerturbedTrial,
        primary_out: GenerationOutput,
        caller_overhead_ms: float,
    ) -> ShadowComparisonRecord:
        candidate_out = self.candidate_pipeline.generate(
            trial.perturbed_query, trial.context_chunks
        )
        prim_eval = AdversarialAbstentionHarness.evaluate_single_trial(trial, primary_out)
        cand_eval = AdversarialAbstentionHarness.evaluate_single_trial(trial, candidate_out)

        prim_cit_prec = (
            prim_eval.citations_grounded / prim_eval.citations_emitted
            if prim_eval.citations_emitted > 0
            else (1.0 if prim_eval.actual_abstain and prim_eval.expected_abstain else 0.0)
        )
        cand_cit_prec = (
            cand_eval.citations_grounded / cand_eval.citations_emitted
            if cand_eval.citations_emitted > 0
            else (1.0 if cand_eval.actual_abstain and cand_eval.expected_abstain else 0.0)
        )

        jaccard = self._compute_answer_jaccard(
            primary_out.answer_text, candidate_out.answer_text
        )
        diffs = self._compute_trajectory_diffs(primary_out, candidate_out)

        return ShadowComparisonRecord(
            trial_id=trial.trial_id,
            query=trial.perturbed_query,
            primary_answer=primary_out.answer_text,
            candidate_answer=candidate_out.answer_text,
            primary_abstained=primary_out.abstained,
            candidate_abstained=candidate_out.abstained,
            primary_fchr_violation=prim_eval.false_confidence_hallucination,
            candidate_fchr_violation=cand_eval.false_confidence_hallucination,
            primary_citation_precision=round(prim_cit_prec, 4),
            candidate_citation_precision=round(cand_cit_prec, 4),
            primary_latency_ms=primary_out.latency_ms,
            candidate_latency_ms=candidate_out.latency_ms,
            primary_caller_overhead_ms=round(caller_overhead_ms, 4),
            primary_cost_usd=primary_out.cost_usd,
            candidate_cost_usd=candidate_out.cost_usd,
            answer_jaccard_similarity=jaccard,
            trajectory_diffs=diffs,
        )

    def route_request(self, trial: PerturbedTrial) -> GenerationOutput:
        """
        Executes Primary Pipeline synchronously and returns immediately.
        If sampled according to `sample_rate`, dispatches Candidate Pipeline
        asynchronously to the background shadow pool.
        """
        self.total_requests += 1
        primary_out = self.primary_pipeline.generate(
            trial.perturbed_query, trial.context_chunks
        )

        t_dispatch_start = time.perf_counter()
        should_shadow = self.rng.random() < self.sample_rate
        if should_shadow:
            overhead_ms = (time.perf_counter() - t_dispatch_start) * 1000.0
            fut = self._executor.submit(
                self._run_shadow_comparison, trial, primary_out, overhead_ms
            )
            self._pending_futures.append(fut)

        return primary_out

    def flush_shadow_queue(self) -> list[ShadowComparisonRecord]:
        """Waits for all in-flight asynchronous shadow tasks to finish and returns records."""
        for fut in self._pending_futures:
            self._completed_records.append(fut.result())
        self._pending_futures.clear()
        return list(self._completed_records)

    @staticmethod
    def paired_bootstrap_ci(
        metric_name: str,
        primary_values: list[float],
        candidate_values: list[float],
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
        is_percentage_point: bool = True,
        lower_is_better: bool = True,
    ) -> BootstrapMetricCI:
        """
        Runs paired bootstrap resampling (`n_iterations=1000`) over paired
        (primary_i, candidate_i) observations to compute the mean delta,
        95% confidence interval, and two-sided bootstrap p-value.
        """
        n = len(primary_values)
        if n == 0 or len(candidate_values) != n:
            raise ValueError("primary_values and candidate_values must be non-empty and equal length")

        mean_p = sum(primary_values) / n
        mean_c = sum(candidate_values) / n
        observed_deltas = [c - p for p, c in zip(primary_values, candidate_values)]
        mean_delta = sum(observed_deltas) / n

        boot_rng = random.Random(seed)
        boot_means: list[float] = []
        non_improving_count = 0

        for _ in range(n_iterations):
            sample_sum = 0.0
            for _ in range(n):
                idx = boot_rng.randrange(n)
                sample_sum += observed_deltas[idx]
            b_mean = sample_sum / n
            boot_means.append(b_mean)
            if lower_is_better:
                if b_mean >= 0.0:
                    non_improving_count += 1
            else:
                if b_mean <= 0.0:
                    non_improving_count += 1

        boot_means.sort()
        alpha = 1.0 - confidence_level
        low_idx = max(0, int(math.floor((alpha / 2.0) * n_iterations)))
        high_idx = min(n_iterations - 1, int(math.ceil((1.0 - alpha / 2.0) * n_iterations)) - 1)

        ci_low = boot_means[low_idx]
        ci_high = boot_means[high_idx]
        half_width = (ci_high - ci_low) / 2.0

        # Two-sided bootstrap p-value bounded by 1 / n_iterations
        raw_p = (2.0 * min(non_improving_count, n_iterations - non_improving_count)) / n_iterations
        p_val = max(1.0 / n_iterations, min(1.0, raw_p))
        sig = p_val < (1.0 - confidence_level)

        if is_percentage_point:
            verb = "reduces" if mean_delta < 0 else "increases"
            summary = (
                f"Candidate {verb} {metric_name} by {mean_delta * 100:+.1f}% "
                f"± {half_width * 100:.1f}% (95% CI [{ci_low * 100:+.1f}%, {ci_high * 100:+.1f}%], "
                f"p = {p_val:.4f})"
            )
        else:
            summary = (
                f"Candidate delta on {metric_name}: {mean_delta:+.4f} "
                f"± {half_width:.4f} (95% CI [{ci_low:+.4f}, {ci_high:+.4f}], p = {p_val:.4f})"
            )

        return BootstrapMetricCI(
            metric_name=metric_name,
            mean_primary=round(mean_p, 5),
            mean_candidate=round(mean_c, 5),
            mean_delta=round(mean_delta, 5),
            ci_lower=round(ci_low, 5),
            ci_upper=round(ci_high, 5),
            half_width=round(half_width, 5),
            p_value=round(p_val, 4),
            is_statistically_significant=sig,
            formatted_summary=summary,
        )

    def build_report(self) -> ShadowRouterReport:
        """Flushes pending shadow tasks and computes full statistical significance report."""
        records = self.flush_shadow_queue()
        if not records:
            raise RuntimeError("No shadow records collected; ensure route_request was called.")

        prim_fchr = [1.0 if r.primary_fchr_violation else 0.0 for r in records]
        cand_fchr = [1.0 if r.candidate_fchr_violation else 0.0 for r in records]

        prim_cit = [r.primary_citation_precision for r in records]
        cand_cit = [r.candidate_citation_precision for r in records]

        prim_lat = [r.primary_latency_ms for r in records]
        cand_lat = [r.candidate_latency_ms for r in records]

        prim_cost = [r.primary_cost_usd for r in records]
        cand_cost = [r.candidate_cost_usd for r in records]

        fchr_ci = self.paired_bootstrap_ci(
            metric_name="False Confidence Hallucination Rate",
            primary_values=prim_fchr,
            candidate_values=cand_fchr,
            n_iterations=self.bootstrap_iterations,
            is_percentage_point=True,
            lower_is_better=True,
        )
        cit_ci = self.paired_bootstrap_ci(
            metric_name="Citation Grounding Precision",
            primary_values=prim_cit,
            candidate_values=cand_cit,
            n_iterations=self.bootstrap_iterations,
            is_percentage_point=True,
            lower_is_better=False,
        )
        lat_ci = self.paired_bootstrap_ci(
            metric_name="End-to-End Latency (ms)",
            primary_values=prim_lat,
            candidate_values=cand_lat,
            n_iterations=self.bootstrap_iterations,
            is_percentage_point=False,
            lower_is_better=True,
        )
        cost_ci = self.paired_bootstrap_ci(
            metric_name="Cost per Query (USD)",
            primary_values=prim_cost,
            candidate_values=cand_cost,
            n_iterations=self.bootstrap_iterations,
            is_percentage_point=False,
            lower_is_better=True,
        )

        agreements = sum(
            1
            for r in records
            if (r.primary_abstained == r.candidate_abstained)
            and (r.primary_fchr_violation == r.candidate_fchr_violation)
        )
        mean_overhead = sum(r.primary_caller_overhead_ms for r in records) / len(records)

        return ShadowRouterReport(
            total_queries_received=self.total_requests,
            shadowed_queries_count=len(records),
            configured_sample_rate=self.sample_rate,
            actual_sample_rate=round(len(records) / max(self.total_requests, 1), 4),
            mean_primary_caller_overhead_ms=round(mean_overhead, 4),
            agreement_rate=round(agreements / len(records), 4),
            fchr_bootstrap=fchr_ci,
            citation_precision_bootstrap=cit_ci,
            latency_ms_bootstrap=lat_ci,
            cost_usd_bootstrap=cost_ci,
            records=records,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
