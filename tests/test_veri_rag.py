"""
Unit & Integration Test Suite for VeriRAG (20 tests).
Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from veri_rag.adversarial_harness import (
    AdversarialAbstentionHarness,
    AdversarialStressMode,
    NaiveRAGGenerator,
    VeriRAGGenerator,
)
from veri_rag.cli import run_all_benchmarks
from veri_rag.dataset_contamination import (
    DatasetContaminationChecker,
    extract_word_ngrams,
    ngram_containment_score,
    ngram_jaccard_similarity,
)
from veri_rag.hybrid_retriever import (
    BM25OkapiIndex,
    Document,
    HashingVectorIndex,
    HybridRRFRetriever,
    MetadataFilter,
    SemanticChunker,
    tokenize,
)
from veri_rag.replay_debugger import (
    CounterfactualReplayDebugger,
    WorkflowStage,
    make_conflict_aware_rrf_filter_node,
    make_strict_entailment_synthesizer_node,
    make_supersession_conflict_check_node,
)
from veri_rag.scenarios import build_enterprise_corpus, build_golden_scenarios
from veri_rag.shadow_router import ShadowTrafficRouter


class TestVeriRAGSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_enterprise_corpus()
        cls.scenarios = build_golden_scenarios()
        cls.retriever = HybridRRFRetriever(cls.corpus)
        cls.harness = AdversarialAbstentionHarness(cls.retriever, cls.scenarios)
        cls.trials = cls.harness.build_stress_trials()

    # -------------------------------------------------------------------------
    # 1. Corpus & Golden Dataset Integrity
    # -------------------------------------------------------------------------
    def test_01_golden_scenarios_count_and_domains(self) -> None:
        self.assertEqual(len(self.scenarios), 30)
        domains = {s.domain for s in self.scenarios}
        self.assertEqual(
            domains,
            {"same_day_delivery", "seller_support", "clinical_protocol"},
        )
        abstain_scenarios = [s for s in self.scenarios if s.should_abstain]
        self.assertEqual(len(abstain_scenarios), 6)

    def test_02_semantic_chunker_preserves_offsets_and_metadata(self) -> None:
        long_text = (
            "Sentence one sets the Sub-Same-Day cutoff at 14:30 local time. "
            "Sentence two specifies cold-chain excursions up to 22 minutes at 4.0°C. "
            "Sentence three restricts hazmat lithium-ion vans to 300 Wh per parcel. "
            "Sentence four caps courier routes at 42 delivery stops and 48 lbs."
        )
        doc = Document(
            doc_id="DOC-TEST-LONG",
            title="Composite Operations Policy",
            content=long_text,
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2026-03-01",
            authority_tier=1,
        )
        chunker = SemanticChunker(max_chars=130, overlap_chars=30)
        chunks = chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 2)
        for ch in chunks:
            self.assertEqual(ch.doc_id, "DOC-TEST-LONG")
            self.assertEqual(ch.region, "US-EAST")
            self.assertGreater(ch.end_char, ch.start_char)
            self.assertIn(ch.text, long_text)

    # -------------------------------------------------------------------------
    # 2. Hybrid Retriever (BM25 + Hashing Vector + RRF + Freshness + Metadata)
    # -------------------------------------------------------------------------
    def test_03_bm25_and_hashing_vector_scoring(self) -> None:
        bm25 = BM25OkapiIndex(self.retriever.chunks)
        vec = HashingVectorIndex(self.retriever.chunks, dim=512)
        q = "Sub-Same-Day evening wave sortation cutoff 14:30"
        lex_scores = bm25.score_subset(q)
        vec_scores = vec.score_subset(q)
        self.assertEqual(len(lex_scores), len(self.retriever.chunks))
        self.assertEqual(len(vec_scores), len(self.retriever.chunks))
        self.assertGreater(max(lex_scores.values()), 0.0)
        self.assertGreater(max(vec_scores.values()), 0.0)

    def test_04_metadata_filter_region_and_superseded(self) -> None:
        f_eu = MetadataFilter(
            region="EU-WEST",
            domain="same_day_delivery",
            exclude_superseded=True,
            allow_global_region_fallback=False,
        )
        results = self.retriever.retrieve(
            "Smart locker dwell time hours",
            top_k=5,
            mode="hybrid_rrf",
            metadata_filter=f_eu,
        )
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertEqual(r.chunk.region, "EU-WEST")
            self.assertEqual(r.chunk.domain, "same_day_delivery")
            self.assertIsNone(r.chunk.superseded_by)

    def test_05_freshness_weighting_ranks_2026_above_2024_legacy(self) -> None:
        results = self.retriever.retrieve(
            "What is the evening wave sortation cutoff time for Sub-Same-Day urban nodes?",
            top_k=3,
            mode="hybrid_rrf",
        )
        ranked_ids = [r.chunk.doc_id for r in results]
        self.assertIn("DOC-SSD-2026-01", ranked_ids)
        if "DOC-SSD-2024-01" in ranked_ids:
            self.assertLess(
                ranked_ids.index("DOC-SSD-2026-01"),
                ranked_ids.index("DOC-SSD-2024-01"),
            )

    def test_06_hybrid_rrf_outperforms_single_modality_baselines(self) -> None:
        lex_m = self.retriever.evaluate_retrieval(self.scenarios, top_k=3, mode="lexical_only")
        vec_m = self.retriever.evaluate_retrieval(self.scenarios, top_k=3, mode="vector_only")
        hyb_m = self.retriever.evaluate_retrieval(self.scenarios, top_k=3, mode="hybrid_rrf")

        self.assertGreater(hyb_m["top1_hit_rate"], lex_m["top1_hit_rate"])
        self.assertGreater(hyb_m["top1_hit_rate"], vec_m["top1_hit_rate"])
        self.assertGreater(hyb_m["mrr_at_k"], lex_m["mrr_at_k"])
        self.assertGreater(hyb_m["fresh_conflict_win_rate"], lex_m["fresh_conflict_win_rate"])

    def test_07_citation_span_extraction_returns_verbatim_substring(self) -> None:
        results = self.retriever.retrieve(
            "QTcF interval threshold milliseconds permanent discontinuation",
            top_k=1,
            mode="hybrid_rrf",
        )
        self.assertEqual(len(results), 1)
        top_res = results[0]
        self.assertEqual(top_res.chunk.doc_id, "DOC-CLN-2026-03")
        self.assertIn(top_res.citation_span.quote, top_res.chunk.text)
        self.assertIn("500 ms", top_res.citation_span.quote)

    # -------------------------------------------------------------------------
    # 3. 5-Mode Adversarial Abstention & Faithfulness Harness
    # -------------------------------------------------------------------------
    def test_08_all_five_adversarial_stress_modes_generated(self) -> None:
        modes_present = {t.stress_mode for t in self.trials}
        self.assertEqual(modes_present, set(AdversarialStressMode))
        self.assertGreaterEqual(len(self.trials), 100)

    def test_09_veri_rag_abstains_on_missing_critical_constraint(self) -> None:
        veri_gen = VeriRAGGenerator()
        missing_trials = [
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.MISSING_CRITICAL_CONSTRAINT
        ]
        self.assertEqual(len(missing_trials), 30)
        abstained_count = sum(
            1
            for t in missing_trials
            if veri_gen.generate(t.perturbed_query, t.context_chunks).abstained
        )
        self.assertEqual(abstained_count, 30)

    def test_10_veri_rag_abstains_on_out_of_distribution_boundary(self) -> None:
        veri_gen = VeriRAGGenerator()
        ood_trials = [
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.OUT_OF_DISTRIBUTION_BOUNDARY
        ]
        self.assertEqual(len(ood_trials), 15)
        for t in ood_trials:
            out = veri_gen.generate(t.perturbed_query, t.context_chunks)
            self.assertTrue(out.abstained)
            self.assertIn("ood_boundary", out.abstention_reason or "")

    def test_11_veri_rag_resolves_contradictory_2024_vs_2026_evidence(self) -> None:
        veri_gen = VeriRAGGenerator()
        conflict_trials = [
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.CONTRADICTORY_EVIDENCE_CONFLICT
        ]
        self.assertEqual(len(conflict_trials), 12)
        for t in conflict_trials:
            out = veri_gen.generate(t.perturbed_query, t.context_chunks)
            ev = self.harness.evaluate_single_trial(t, out)
            self.assertTrue(ev.conflict_resolved_correctly)
            self.assertFalse(ev.false_confidence_hallucination)

    def test_12_veri_rag_rejects_citation_spoofing_probes(self) -> None:
        veri_gen = VeriRAGGenerator()
        spoof_trials = [
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.CITATION_SPOOFING_PROBE
        ]
        for t in spoof_trials:
            out = veri_gen.generate(t.perturbed_query, t.context_chunks)
            self.assertNotIn(t.spoof_decoy_doc_id, out.cited_doc_ids)

    def test_13_harness_aggregate_metrics_improvement(self) -> None:
        naive_summary, _ = self.harness.evaluate_pipeline(
            NaiveRAGGenerator(), self.trials
        )
        veri_summary, _ = self.harness.evaluate_pipeline(
            VeriRAGGenerator(), self.trials
        )
        self.assertGreater(
            veri_summary.abstention_precision,
            naive_summary.abstention_precision + 0.80,
        )
        self.assertGreater(
            veri_summary.abstention_recall,
            naive_summary.abstention_recall + 0.80,
        )
        self.assertLess(
            veri_summary.false_confidence_hallucination_rate,
            naive_summary.false_confidence_hallucination_rate - 0.50,
        )
        self.assertEqual(veri_summary.conflict_resolution_accuracy, 1.0)

    # -------------------------------------------------------------------------
    # 4. Shadow Traffic Router & Paired Bootstrap Engine
    # -------------------------------------------------------------------------
    def test_14_shadow_router_5pct_sampling_and_non_blocking_overhead(self) -> None:
        router = ShadowTrafficRouter(
            primary_pipeline=NaiveRAGGenerator(),
            candidate_pipeline=VeriRAGGenerator(),
            sample_rate=0.05,
            random_seed=101,
            bootstrap_iterations=200,
        )
        for i in range(300):
            router.route_request(self.trials[i % len(self.trials)])
        report = router.build_report()
        router.shutdown()

        self.assertEqual(report.total_queries_received, 300)
        self.assertGreaterEqual(report.shadowed_queries_count, 5)
        self.assertLessEqual(report.shadowed_queries_count, 35)
        self.assertLess(report.mean_primary_caller_overhead_ms, 5.0)

    def test_15_paired_bootstrap_statistical_significance(self) -> None:
        ci = ShadowTrafficRouter.paired_bootstrap_ci(
            metric_name="False Confidence Hallucination Rate",
            primary_values=[1.0] * 35 + [0.0] * 15,
            candidate_values=[0.0] * 48 + [1.0] * 2,
            n_iterations=1000,
            confidence_level=0.95,
            seed=42,
            is_percentage_point=True,
            lower_is_better=True,
        )
        self.assertLess(ci.mean_delta, -0.50)
        self.assertLess(ci.ci_upper, -0.45)
        self.assertTrue(ci.is_statistically_significant)
        self.assertIn("Candidate reduces False Confidence Hallucination Rate", ci.formatted_summary)

    # -------------------------------------------------------------------------
    # 5. Counterfactual Trajectory Replay Debugger
    # -------------------------------------------------------------------------
    def test_16_replay_debugger_records_5_stage_snapshots(self) -> None:
        debugger = CounterfactualReplayDebugger()
        conflict_trial = next(
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.CONTRADICTORY_EVIDENCE_CONFLICT
        )
        traj = debugger.record_trajectory(conflict_trial)
        self.assertEqual(len(traj.snapshots), 5)
        self.assertEqual(
            [s.stage for s in traj.snapshots],
            [
                WorkflowStage.QUERY_REWRITE,
                WorkflowStage.HYBRID_RETRIEVE,
                WorkflowStage.RERANK_FILTER,
                WorkflowStage.CONFLICT_CHECK,
                WorkflowStage.SYNTHESIZE_WITH_CITATIONS,
            ],
        )
        self.assertEqual(traj.defect_category, "STALE_POLICY_CONFLICT_HALLUCINATION")

    def test_17_replay_debugger_swaps_single_node_and_reuses_upstream_state(self) -> None:
        debugger = CounterfactualReplayDebugger()
        conflict_trial = next(
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.CONTRADICTORY_EVIDENCE_CONFLICT
        )
        traj = debugger.record_trajectory(conflict_trial)

        # Swap stage 4 (CONFLICT_CHECK) while reusing stages 1, 2, 3 from recorded state
        diff = debugger.replay_with_node_swap(
            traj, make_supersession_conflict_check_node()
        )
        self.assertEqual(diff.intervened_stage, WorkflowStage.CONFLICT_CHECK)
        self.assertEqual(
            diff.reused_upstream_stages,
            ["QUERY_REWRITE", "HYBRID_RETRIEVE", "RERANK_FILTER"],
        )
        self.assertEqual(
            diff.reexecuted_downstream_stages,
            ["CONFLICT_CHECK", "SYNTHESIZE_WITH_CITATIONS"],
        )
        self.assertTrue(diff.defect_resolved)
        self.assertEqual(diff.counterfactual_defect_category, "FAITHFUL")

    def test_18_replay_debugger_isolates_missing_constraint_at_synthesizer(self) -> None:
        debugger = CounterfactualReplayDebugger()
        missing_trial = next(
            t
            for t in self.trials
            if t.stress_mode == AdversarialStressMode.MISSING_CRITICAL_CONSTRAINT
        )
        traj = debugger.record_trajectory(missing_trial)
        self.assertEqual(
            traj.defect_category,
            "FALSE_CONFIDENCE_ON_MISSING_OR_OOD_CONSTRAINT",
        )
        diff = debugger.replay_with_node_swap(
            traj, make_strict_entailment_synthesizer_node()
        )
        self.assertTrue(diff.defect_resolved)
        self.assertTrue(diff.counterfactual_abstained)

    # -------------------------------------------------------------------------
    # 6. Dataset Contamination Checker & CLI End-to-End
    # -------------------------------------------------------------------------
    def test_19_dataset_contamination_checker_flags_injected_leak(self) -> None:
        checker = DatasetContaminationChecker()
        clean_2026_docs = [d for d in self.corpus if d.superseded_by is None]
        clean_report = checker.scan(self.scenarios, clean_2026_docs)
        self.assertEqual(clean_report.contaminated_leak_count, 0)

        leaked_doc = Document(
            doc_id="LEAK-DOC-999",
            title="Leaked Eval Benchmark Entry",
            content=f"{self.scenarios[0].query} {self.scenarios[0].ground_truth_answer}",
            domain="same_day_delivery",
        )
        leak_report = checker.scan(self.scenarios, clean_2026_docs + [leaked_doc])
        self.assertGreaterEqual(leak_report.contaminated_leak_count, 1)
        self.assertIn("SCN-SSD-01", leak_report.flagged_scenario_ids)

    def test_20_cli_end_to_end_json_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "results.json"
            payload = run_all_benchmarks(out_file)
            self.assertTrue(out_file.exists())
            self.assertIn("retrieval_benchmark", payload)
            self.assertIn("adversarial_harness", payload)
            self.assertIn("shadow_routing", payload)
            self.assertIn("counterfactual_replay_debugger", payload)
            self.assertIn("dataset_contamination_audit", payload)


if __name__ == "__main__":
    unittest.main()
