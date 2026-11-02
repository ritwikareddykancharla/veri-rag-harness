"""
CLI Entry Point (`python3 -m veri_rag.cli`) for VeriRAG.

Executes:
  1. Hybrid Retrieval Benchmark (BM25Okapi vs. Dense Hashing Vector vs. RRF vs. Freshness-Aware Hybrid RRF)
  2. 5-Mode Adversarial Abstention & Faithfulness Evaluation Harness (111 trials across 30 scenarios)
  3. Asynchronous Shadow Traffic Router & 1,000-Iteration Paired Bootstrap Significance Engine
  4. Counterfactual Trajectory Replay Debugger (Single-Node Root-Cause Isolation)
  5. Dataset Contamination & Benchmark Leakage Audit
  6. Persists structured results to `benchmarks/latest_results.json`
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from veri_rag.adversarial_harness import (
    AdversarialAbstentionHarness,
    AdversarialStressMode,
    NaiveRAGGenerator,
    VeriRAGGenerator,
)
from veri_rag.dataset_contamination import DatasetContaminationChecker
from veri_rag.hybrid_retriever import Document, HybridRRFRetriever
from veri_rag.replay_debugger import CounterfactualReplayDebugger
from veri_rag.scenarios import build_enterprise_corpus, build_golden_scenarios
from veri_rag.shadow_router import ShadowTrafficRouter


def run_all_benchmarks(output_path: Path) -> dict[str, Any]:
    corpus = build_enterprise_corpus()
    scenarios = build_golden_scenarios()
    retriever = HybridRRFRetriever(corpus)

    # -------------------------------------------------------------------------
    # 1. Retrieval Benchmark across 4 modes
    # -------------------------------------------------------------------------
    retrieval_modes = [
        "lexical_only",
        "vector_only",
        "hybrid_rrf_no_freshness",
        "hybrid_rrf",
    ]
    retrieval_benchmarks: dict[str, dict[str, float]] = {}
    for mode in retrieval_modes:
        retrieval_benchmarks[mode] = retriever.evaluate_retrieval(
            scenarios, top_k=3, mode=mode, use_region_filter=True
        )

    # -------------------------------------------------------------------------
    # 2. 5-Mode Adversarial Abstention & Faithfulness Harness
    # -------------------------------------------------------------------------
    harness = AdversarialAbstentionHarness(retriever, scenarios)
    trials = harness.build_stress_trials()

    baseline_gen = NaiveRAGGenerator()
    candidate_gen = VeriRAGGenerator()

    baseline_summary, _ = harness.evaluate_pipeline(baseline_gen, trials)
    candidate_summary, _ = harness.evaluate_pipeline(candidate_gen, trials)

    # -------------------------------------------------------------------------
    # 3. Shadow Traffic Router & Paired Bootstrap Significance Engine
    # -------------------------------------------------------------------------
    # 3a. Full paired benchmark evaluation (100% shadow mirror, 1,000 bootstrap iterations)
    full_shadow_router = ShadowTrafficRouter(
        primary_pipeline=baseline_gen,
        candidate_pipeline=candidate_gen,
        sample_rate=1.0,
        random_seed=42,
        bootstrap_iterations=1000,
    )
    for t in trials:
        full_shadow_router.route_request(t)
    full_shadow_report = full_shadow_router.build_report()
    full_shadow_router.shutdown()

    # 3b. Production 5% shadow traffic simulation over 400 synthetic live requests
    prod_shadow_router = ShadowTrafficRouter(
        primary_pipeline=baseline_gen,
        candidate_pipeline=candidate_gen,
        sample_rate=0.05,
        random_seed=101,
        bootstrap_iterations=1000,
    )
    for i in range(400):
        prod_shadow_router.route_request(trials[i % len(trials)])
    prod_shadow_report = prod_shadow_router.build_report()
    prod_shadow_router.shutdown()

    # -------------------------------------------------------------------------
    # 4. Counterfactual Trajectory Replay Debugger
    # -------------------------------------------------------------------------
    debugger = CounterfactualReplayDebugger()

    def _find_failing_trial(mode: AdversarialStressMode) -> Any:
        for candidate_t in trials:
            if candidate_t.stress_mode == mode:
                rec = debugger.record_trajectory(candidate_t)
                if rec.defect_category != "FAITHFUL":
                    return candidate_t
        return next(t for t in trials if t.stress_mode == mode)

    conflict_trial = _find_failing_trial(
        AdversarialStressMode.CONTRADICTORY_EVIDENCE_CONFLICT
    )
    distractor_trial = _find_failing_trial(
        AdversarialStressMode.IRRELEVANT_DISTRACTOR_INJECTION
    )
    missing_trial = _find_failing_trial(
        AdversarialStressMode.MISSING_CRITICAL_CONSTRAINT
    )
    spoof_trial = _find_failing_trial(
        AdversarialStressMode.CITATION_SPOOFING_PROBE
    )

    replay_case_studies: list[dict[str, Any]] = []
    for sample_trial in (
        conflict_trial,
        distractor_trial,
        missing_trial,
        spoof_trial,
    ):
        rec_traj = debugger.record_trajectory(sample_trial)
        diffs = debugger.isolate_root_cause(rec_traj)
        winning_diff = next((d for d in diffs if d.defect_resolved), diffs[-1])
        replay_case_studies.append(
            {
                "trial_id": sample_trial.trial_id,
                "stress_mode": sample_trial.stress_mode.value,
                "original_defect": rec_traj.defect_category,
                "earliest_resolving_stage": winning_diff.intervened_stage.value,
                "original_node": winning_diff.original_impl_name,
                "replacement_node": winning_diff.replacement_impl_name,
                "reused_upstream_stages": winning_diff.reused_upstream_stages,
                "reexecuted_downstream_stages": winning_diff.reexecuted_downstream_stages,
                "original_cited_docs": winning_diff.original_cited_docs,
                "counterfactual_cited_docs": winning_diff.counterfactual_cited_docs,
                "defect_resolved": winning_diff.defect_resolved,
                "causal_attribution_verdict": winning_diff.causal_attribution_verdict,
            }
        )

    # -------------------------------------------------------------------------
    # 5. Dataset Contamination Audit
    # -------------------------------------------------------------------------
    checker = DatasetContaminationChecker()
    # Scan against active 2026 Tier-1/Tier-2 corpus (excluding the legacy 2024 doc with injected question)
    clean_2026_corpus = [d for d in corpus if d.superseded_by is None]
    clean_scan_report = checker.scan(scenarios, clean_2026_corpus)

    # Scan against full corpus + synthetic SFT training leak doc to verify leak detection
    leaked_training_split = list(corpus) + [
        Document(
            doc_id="TRAIN-LEAK-SFT-099",
            title="Synthetic Fine-Tuning Q&A Dump",
            content=(
                f"Question: {scenarios[10].query} "
                f"Answer: {scenarios[10].ground_truth_answer}"
            ),
            domain="seller_support",
            region="US-EAST",
            effective_date="2026-06-01",
            authority_tier=3,
        )
    ]
    leaked_scan_report = checker.scan(scenarios, leaked_training_split)

    # -------------------------------------------------------------------------
    # Assemble & Persist JSON Payload
    # -------------------------------------------------------------------------
    payload: dict[str, Any] = {
        "corpus_summary": {
            "total_documents": len(corpus),
            "total_chunks": len(retriever.chunks),
            "golden_scenarios": len(scenarios),
            "adversarial_trials": len(trials),
        },
        "retrieval_benchmark": retrieval_benchmarks,
        "adversarial_harness": {
            "baseline_naive_rag": asdict(baseline_summary),
            "candidate_veri_rag": asdict(candidate_summary),
        },
        "shadow_routing": {
            "full_benchmark_100pct": {
                "total_queries": full_shadow_report.total_queries_received,
                "shadowed_queries": full_shadow_report.shadowed_queries_count,
                "mean_primary_caller_overhead_ms": full_shadow_report.mean_primary_caller_overhead_ms,
                "agreement_rate": full_shadow_report.agreement_rate,
                "fchr_bootstrap": asdict(full_shadow_report.fchr_bootstrap),
                "citation_precision_bootstrap": asdict(
                    full_shadow_report.citation_precision_bootstrap
                ),
                "latency_ms_bootstrap": asdict(full_shadow_report.latency_ms_bootstrap),
                "cost_usd_bootstrap": asdict(full_shadow_report.cost_usd_bootstrap),
            },
            "production_simulation_5pct": {
                "total_queries": prod_shadow_report.total_queries_received,
                "shadowed_queries": prod_shadow_report.shadowed_queries_count,
                "configured_sample_rate": prod_shadow_report.configured_sample_rate,
                "actual_sample_rate": prod_shadow_report.actual_sample_rate,
                "mean_primary_caller_overhead_ms": prod_shadow_report.mean_primary_caller_overhead_ms,
                "fchr_bootstrap_summary": prod_shadow_report.fchr_bootstrap.formatted_summary,
            },
        },
        "counterfactual_replay_debugger": {
            "case_studies": replay_case_studies,
        },
        "dataset_contamination_audit": {
            "active_2026_corpus_scan": {
                "total_scenarios": clean_scan_report.total_scenarios_scanned,
                "clean_count": clean_scan_report.clean_count,
                "borderline_count": clean_scan_report.borderline_count,
                "contaminated_leak_count": clean_scan_report.contaminated_leak_count,
                "contamination_rate": clean_scan_report.contamination_rate,
                "flagged_scenario_ids": clean_scan_report.flagged_scenario_ids,
            },
            "leaked_sft_split_scan": {
                "total_scenarios": leaked_scan_report.total_scenarios_scanned,
                "clean_count": leaked_scan_report.clean_count,
                "borderline_count": leaked_scan_report.borderline_count,
                "contaminated_leak_count": leaked_scan_report.contaminated_leak_count,
                "contamination_rate": leaked_scan_report.contamination_rate,
                "flagged_scenario_ids": leaked_scan_report.flagged_scenario_ids,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def print_summary(payload: dict[str, Any], output_path: Path) -> None:
    print("=" * 88)
    print("VeriRAG: Hybrid Retrieval, Adversarial Abstention, Shadow Router & Replay Benchmark")
    print("=" * 88)

    cs = payload["corpus_summary"]
    print(
        f"Corpus: {cs['total_documents']} docs ({cs['total_chunks']} semantic chunks) | "
        f"Golden Scenarios: {cs['golden_scenarios']} | Adversarial Stress Trials: {cs['adversarial_trials']}"
    )
    print("-" * 88)
    print("1. HYBRID RETRIEVAL ENGINE BENCHMARK (Top-K = 3)")
    print(
        f"{'Retrieval Mode':<28} {'Recall@3':>10} {'Top-1 Hit':>11} {'MRR@3':>10} {'nDCG@3':>10} {'FreshWin%':>11}"
    )
    for mode, metrics in payload["retrieval_benchmark"].items():
        print(
            f"{mode:<28} "
            f"{metrics['recall_at_k'] * 100:>9.1f}% "
            f"{metrics['top1_hit_rate'] * 100:>10.1f}% "
            f"{metrics['mrr_at_k']:>10.4f} "
            f"{metrics['ndcg_at_k']:>10.4f} "
            f"{metrics['fresh_conflict_win_rate'] * 100:>10.1f}%"
        )

    print("-" * 88)
    print("2. 5-MODE ADVERSARIAL ABSTENTION & FAITHFULNESS HARNESS")
    base = payload["adversarial_harness"]["baseline_naive_rag"]
    cand = payload["adversarial_harness"]["candidate_veri_rag"]
    print(
        f"{'Metric':<38} {'NaiveTopK (Baseline)':>22} {'VeriRAG (Candidate)':>22}"
    )
    print(
        f"{'Abstention Precision':<38} {base['abstention_precision'] * 100:>21.1f}% {cand['abstention_precision'] * 100:>21.1f}%"
    )
    print(
        f"{'Abstention Recall':<38} {base['abstention_recall'] * 100:>21.1f}% {cand['abstention_recall'] * 100:>21.1f}%"
    )
    print(
        f"{'False Confidence Hallucination (FCHR)':<38} {base['false_confidence_hallucination_rate'] * 100:>21.1f}% {cand['false_confidence_hallucination_rate'] * 100:>21.1f}%"
    )
    print(
        f"{'Citation Grounding Precision':<38} {base['citation_grounding_precision'] * 100:>21.1f}% {cand['citation_grounding_precision'] * 100:>21.1f}%"
    )
    print(
        f"{'Conflict Resolution Accuracy (2026>24)':<38} {base['conflict_resolution_accuracy'] * 100:>21.1f}% {cand['conflict_resolution_accuracy'] * 100:>21.1f}%"
    )

    print("-" * 88)
    print("3. SHADOW TRAFFIC ROUTER & PAIRED BOOTSTRAP SIGNIFICANCE (1,000 Iterations)")
    sr = payload["shadow_routing"]["full_benchmark_100pct"]
    prod = payload["shadow_routing"]["production_simulation_5pct"]
    print(f"  * {sr['fchr_bootstrap']['formatted_summary']}")
    print(f"  * {sr['citation_precision_bootstrap']['formatted_summary']}")
    print(
        f"  * 5% Prod Shadow Simulation: {prod['shadowed_queries']}/{prod['total_queries']} "
        f"queries mirrored ({prod['actual_sample_rate'] * 100:.1f}%), "
        f"primary caller overhead = {prod['mean_primary_caller_overhead_ms']:.4f} ms"
    )

    print("-" * 88)
    print("4. COUNTERFACTUAL TRAJECTORY REPLAY DEBUGGER (Single-Node Root Cause Isolation)")
    for case in payload["counterfactual_replay_debugger"]["case_studies"]:
        print(f"  * [{case['trial_id']}] {case['causal_attribution_verdict']}")

    print("-" * 88)
    print("5. DATASET CONTAMINATION & BENCHMARK LEAKAGE AUDIT")
    c_clean = payload["dataset_contamination_audit"]["active_2026_corpus_scan"]
    c_leak = payload["dataset_contamination_audit"]["leaked_sft_split_scan"]
    print(
        f"  * Active 2026 Retrieval Corpus: {c_clean['contaminated_leak_count']}/{c_clean['total_scenarios']} leaked "
        f"(contamination rate = {c_clean['contamination_rate'] * 100:.1f}%)"
    )
    print(
        f"  * Unfiltered Corpus + SFT Leak Split: {c_leak['contaminated_leak_count']}/{c_leak['total_scenarios']} leaked "
        f"-> Flagged scenarios: {c_leak['flagged_scenario_ids']}"
    )
    print("=" * 88)
    print(f"Saved full benchmark artifact to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VeriRAG evaluation harness, shadow router, and counterfactual replay benchmarks."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/latest_results.json",
        help="Path to write JSON benchmark report.",
    )
    args = parser.parse_args()
    out_path = Path(args.output)
    payload = run_all_benchmarks(out_path)
    print_summary(payload, out_path)


if __name__ == "__main__":
    main()
