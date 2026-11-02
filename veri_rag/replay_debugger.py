"""
Counterfactual Trajectory Replay Debugger for Multi-Hop RAG & Agent Workflows.

Records every node in the 5-stage RAG execution graph:
  QUERY_REWRITE -> HYBRID_RETRIEVE -> RERANK_FILTER -> CONFLICT_CHECK -> SYNTHESIZE_WITH_CITATIONS

Allows swapping out a single node (e.g., replacing `NaiveTopKFilter` with
`ConflictAwareRRF` at `RERANK_FILTER`, or swapping the prompt template at
`SYNTHESIZE_WITH_CITATIONS`) and replaying the remainder of the trajectory
from recorded upstream state to isolate the exact causal step behind a
hallucination, stale policy citation, or false confidence failure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from typing import Any, Callable, Optional

from veri_rag.adversarial_harness import (
    AdversarialAbstentionHarness,
    GenerationOutput,
    NaiveRAGGenerator,
    PerturbedTrial,
    TrialEvaluationResult,
    VeriRAGGenerator,
)
from veri_rag.hybrid_retriever import DocumentChunk, HybridRRFRetriever


class WorkflowStage(StrEnum):
    """Canonical 5-stage multi-hop RAG workflow DAG."""

    QUERY_REWRITE = "QUERY_REWRITE"
    HYBRID_RETRIEVE = "HYBRID_RETRIEVE"
    RERANK_FILTER = "RERANK_FILTER"
    CONFLICT_CHECK = "CONFLICT_CHECK"
    SYNTHESIZE_WITH_CITATIONS = "SYNTHESIZE_WITH_CITATIONS"


STAGE_ORDER: tuple[WorkflowStage, ...] = (
    WorkflowStage.QUERY_REWRITE,
    WorkflowStage.HYBRID_RETRIEVE,
    WorkflowStage.RERANK_FILTER,
    WorkflowStage.CONFLICT_CHECK,
    WorkflowStage.SYNTHESIZE_WITH_CITATIONS,
)


def _hash_state(state: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 fingerprint of a workflow state dictionary."""
    serializable = {
        "rewritten_query": state.get("rewritten_query", ""),
        "chunk_ids": [
            c.chunk_id if isinstance(c, DocumentChunk) else str(c)
            for c in state.get("active_chunks", [])
        ],
        "conflict_detected": bool(state.get("conflict_detected", False)),
        "dropped_docs": list(state.get("dropped_docs", [])),
        "abstained": (
            state["generation_output"].abstained
            if isinstance(state.get("generation_output"), GenerationOutput)
            else None
        ),
        "cited_docs": (
            state["generation_output"].cited_doc_ids
            if isinstance(state.get("generation_output"), GenerationOutput)
            else []
        ),
    }
    payload = json.dumps(serializable, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class WorkflowNode:
    """Named executable node for a specific WorkflowStage."""

    stage: WorkflowStage
    impl_name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class NodeExecutionSnapshot:
    """Recorded snapshot of a single stage's execution in a trajectory."""

    stage: WorkflowStage
    impl_name: str
    input_state_hash: str
    output_state_hash: str
    input_state: dict[str, Any]
    output_state: dict[str, Any]
    active_doc_ids_after: list[str]


@dataclass(frozen=True)
class RecordedTrajectory:
    """Full recorded 5-stage workflow trajectory for a PerturbedTrial."""

    trajectory_id: str
    trial: PerturbedTrial
    snapshots: list[NodeExecutionSnapshot]
    final_output: GenerationOutput
    evaluation: TrialEvaluationResult
    defect_category: str


@dataclass(frozen=True)
class CounterfactualReplayDiff:
    """Result of swapping a single node and replaying downstream stages."""

    trajectory_id: str
    intervened_stage: WorkflowStage
    original_impl_name: str
    replacement_impl_name: str
    reused_upstream_stages: list[str]
    reexecuted_downstream_stages: list[str]
    original_defect_category: str
    counterfactual_defect_category: str
    original_cited_docs: list[str]
    counterfactual_cited_docs: list[str]
    original_abstained: bool
    counterfactual_abstained: bool
    defect_resolved: bool
    causal_attribution_verdict: str
    replayed_trajectory: RecordedTrajectory


# =============================================================================
# Built-in Swappable Node Implementations (Baseline vs. Guarded Counterfactuals)
# =============================================================================


def make_identity_query_rewrite_node() -> WorkflowNode:
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        new_state["rewritten_query"] = state["raw_query"]
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.QUERY_REWRITE,
        impl_name="IdentityQueryPassthrough",
        fn=_run,
    )


def make_acronym_expanding_query_rewrite_node() -> WorkflowNode:
    expansions = {
        "ssd": "Sub-Same-Day SSD",
        "mfc": "Micro-Fulfillment Center MFC",
        "sfp": "Seller-Fulfilled Prime SFP",
        "anc": "Absolute Neutrophil Count ANC",
        "icf": "Informed Consent Form ICF",
        "ecoa": "Electronic Clinical Outcome Assessment eCOA",
    }

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        q = state["raw_query"]
        q_lower = q.lower()
        added: list[str] = []
        for acr, full in expansions.items():
            if acr in q_lower and full.lower() not in q_lower:
                added.append(full)
        new_state["rewritten_query"] = f"{q} ({' '.join(added)})" if added else q
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.QUERY_REWRITE,
        impl_name="DomainAcronymQueryRewriter",
        fn=_run,
    )


def make_trial_context_retrieve_node() -> WorkflowNode:
    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        new_state["active_chunks"] = list(state["trial"].context_chunks)
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.HYBRID_RETRIEVE,
        impl_name="RecordedContextRetriever",
        fn=_run,
    )


def make_naive_topk_filter_node(top_k: int = 3) -> WorkflowNode:
    """Baseline RERANK_FILTER node that preserves initial chunk order without freshness/RRF."""

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        chunks: list[DocumentChunk] = list(state.get("active_chunks", []))
        new_state["active_chunks"] = chunks[:top_k]
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.RERANK_FILTER,
        impl_name="NaiveTopKFilter",
        fn=_run,
    )


def make_conflict_aware_rrf_filter_node(top_k: int = 3) -> WorkflowNode:
    """
    Counterfactual RERANK_FILTER node (`ConflictAwareRRF`) that re-scores candidate
    chunks using query lexical/entity overlap + freshness & authority tier weighting,
    demoting stale 2024 Tier-3 chunks and irrelevant distractors.
    """

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        query = state.get("rewritten_query", state["raw_query"])
        chunks: list[DocumentChunk] = list(state.get("active_chunks", []))

        def _score_chunk(ch: DocumentChunk) -> float:
            span = HybridRRFRetriever.extract_citation_span(ch, query)
            fresh = HybridRRFRetriever.compute_freshness_multiplier(
                ch, freshness_weight=0.35
            )
            spoof_penalty = 0.1 if ch.metadata.get("is_spoof_decoy", False) else 1.0
            return (span.overlap_score + 0.25) * fresh * spoof_penalty

        ranked = sorted(chunks, key=_score_chunk, reverse=True)
        new_state["active_chunks"] = ranked[:top_k]
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.RERANK_FILTER,
        impl_name="ConflictAwareRRF",
        fn=_run,
    )


def make_noop_conflict_check_node() -> WorkflowNode:
    """Baseline CONFLICT_CHECK node that performs no supersession filtering."""

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        new_state["conflict_detected"] = False
        new_state["dropped_docs"] = []
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.CONFLICT_CHECK,
        impl_name="UncheckedConflictPassthrough",
        fn=_run,
    )


def make_supersession_conflict_check_node() -> WorkflowNode:
    """Counterfactual CONFLICT_CHECK node that drops superseded 2024 and spoof decoy docs."""

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        chunks: list[DocumentChunk] = list(state.get("active_chunks", []))
        clean_chunks, conflict_detected, dropped = VeriRAGGenerator._resolve_conflicts(
            chunks
        )
        new_state["active_chunks"] = clean_chunks
        new_state["conflict_detected"] = conflict_detected
        new_state["dropped_docs"] = dropped
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.CONFLICT_CHECK,
        impl_name="SupersessionConflictResolver",
        fn=_run,
    )


def make_permissive_synthesizer_node() -> WorkflowNode:
    """Baseline SYNTHESIZE_WITH_CITATIONS node using NaiveRAGGenerator."""
    naive_gen = NaiveRAGGenerator()

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        query = state.get("rewritten_query", state["raw_query"])
        chunks: list[DocumentChunk] = list(state.get("active_chunks", []))
        out = naive_gen.generate(query, chunks)
        new_state["generation_output"] = out
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.SYNTHESIZE_WITH_CITATIONS,
        impl_name="PermissivePromptSynthesizer",
        fn=_run,
    )


def make_strict_entailment_synthesizer_node() -> WorkflowNode:
    """Counterfactual SYNTHESIZE_WITH_CITATIONS node with abstention & span verification."""
    veri_gen = VeriRAGGenerator()

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        new_state = copy.copy(state)
        query = state.get("rewritten_query", state["raw_query"])
        chunks: list[DocumentChunk] = list(state.get("active_chunks", []))
        out = veri_gen.generate(query, chunks)
        new_state["generation_output"] = out
        return new_state

    return WorkflowNode(
        stage=WorkflowStage.SYNTHESIZE_WITH_CITATIONS,
        impl_name="StrictCitationEntailmentSynthesizer",
        fn=_run,
    )


class CounterfactualReplayDebugger:
    """
    Records multi-hop RAG trajectories and executes single-node counterfactual
    replays from saved intermediate snapshots to attribute failure root causes.
    """

    @staticmethod
    def classify_defect(
        trial: PerturbedTrial,
        output: GenerationOutput,
        ev: TrialEvaluationResult,
    ) -> str:
        if not ev.false_confidence_hallucination and (
            ev.expected_abstain == ev.actual_abstain
        ):
            return "FAITHFUL"
        if ev.expected_abstain and not ev.actual_abstain:
            return "FALSE_CONFIDENCE_ON_MISSING_OR_OOD_CONSTRAINT"
        ans_lower = output.answer_text.lower()
        if any(s.lower() in ans_lower for s in trial.forbidden_stale_facts):
            return "STALE_POLICY_CONFLICT_HALLUCINATION"
        if trial.spoof_decoy_doc_id and trial.spoof_decoy_doc_id in output.cited_doc_ids:
            return "SPOOFED_CITATION_HALLUCINATION"
        return "UNGROUNDED_DISTRACTOR_HALLUCINATION"

    def record_trajectory(
        self,
        trial: PerturbedTrial,
        nodes_by_stage: Optional[dict[WorkflowStage, WorkflowNode]] = None,
    ) -> RecordedTrajectory:
        """Executes all 5 stages in order and records state snapshots at every node."""
        pipeline_nodes = nodes_by_stage or {
            WorkflowStage.QUERY_REWRITE: make_identity_query_rewrite_node(),
            WorkflowStage.HYBRID_RETRIEVE: make_trial_context_retrieve_node(),
            WorkflowStage.RERANK_FILTER: make_naive_topk_filter_node(),
            WorkflowStage.CONFLICT_CHECK: make_noop_conflict_check_node(),
            WorkflowStage.SYNTHESIZE_WITH_CITATIONS: make_permissive_synthesizer_node(),
        }

        state: dict[str, Any] = {
            "trial": trial,
            "raw_query": trial.perturbed_query,
            "rewritten_query": trial.perturbed_query,
            "active_chunks": [],
            "conflict_detected": False,
            "dropped_docs": [],
            "generation_output": None,
        }

        snapshots: list[NodeExecutionSnapshot] = []
        for stage in STAGE_ORDER:
            node = pipeline_nodes[stage]
            in_snapshot = copy.copy(state)
            in_hash = _hash_state(in_snapshot)
            out_state = node.fn(in_snapshot)
            out_hash = _hash_state(out_state)
            active_docs = [c.doc_id for c in out_state.get("active_chunks", [])]
            snapshots.append(
                NodeExecutionSnapshot(
                    stage=stage,
                    impl_name=node.impl_name,
                    input_state_hash=in_hash,
                    output_state_hash=out_hash,
                    input_state=in_snapshot,
                    output_state=copy.copy(out_state),
                    active_doc_ids_after=active_docs,
                )
            )
            state = out_state

        final_out: GenerationOutput = state["generation_output"]
        ev = AdversarialAbstentionHarness.evaluate_single_trial(trial, final_out)
        defect = self.classify_defect(trial, final_out, ev)

        return RecordedTrajectory(
            trajectory_id=f"TRJ::{trial.trial_id}",
            trial=trial,
            snapshots=snapshots,
            final_output=final_out,
            evaluation=ev,
            defect_category=defect,
        )

    def replay_with_node_swap(
        self,
        recorded: RecordedTrajectory,
        replacement_node: WorkflowNode,
        downstream_overrides: Optional[dict[WorkflowStage, WorkflowNode]] = None,
    ) -> CounterfactualReplayDiff:
        """
        Replays `recorded` trajectory starting at `replacement_node.stage` using the
        exact recorded input state of that stage (without re-running earlier nodes).
        """
        target_stage = replacement_node.stage
        target_idx = STAGE_ORDER.index(target_stage)

        reused_snapshots = list(recorded.snapshots[:target_idx])
        reused_stage_names = [s.stage.value for s in reused_snapshots]

        # Retrieve the recorded input state right before `target_stage`
        state = copy.copy(recorded.snapshots[target_idx].input_state)

        # Build node map for remaining stages from original snapshots unless overridden
        default_nodes: dict[WorkflowStage, WorkflowNode] = {
            WorkflowStage.QUERY_REWRITE: make_identity_query_rewrite_node(),
            WorkflowStage.HYBRID_RETRIEVE: make_trial_context_retrieve_node(),
            WorkflowStage.RERANK_FILTER: make_naive_topk_filter_node(),
            WorkflowStage.CONFLICT_CHECK: make_noop_conflict_check_node(),
            WorkflowStage.SYNTHESIZE_WITH_CITATIONS: make_permissive_synthesizer_node(),
        }
        if downstream_overrides:
            default_nodes.update(downstream_overrides)
        default_nodes[target_stage] = replacement_node

        new_snapshots: list[NodeExecutionSnapshot] = list(reused_snapshots)
        reexecuted_stage_names: list[str] = []

        for stage in STAGE_ORDER[target_idx:]:
            node = default_nodes[stage]
            reexecuted_stage_names.append(stage.value)
            in_snapshot = copy.copy(state)
            in_hash = _hash_state(in_snapshot)
            out_state = node.fn(in_snapshot)
            out_hash = _hash_state(out_state)
            active_docs = [c.doc_id for c in out_state.get("active_chunks", [])]
            new_snapshots.append(
                NodeExecutionSnapshot(
                    stage=stage,
                    impl_name=node.impl_name,
                    input_state_hash=in_hash,
                    output_state_hash=out_hash,
                    input_state=in_snapshot,
                    output_state=copy.copy(out_state),
                    active_doc_ids_after=active_docs,
                )
            )
            state = out_state

        cf_out: GenerationOutput = state["generation_output"]
        cf_eval = AdversarialAbstentionHarness.evaluate_single_trial(
            recorded.trial, cf_out
        )
        cf_defect = self.classify_defect(recorded.trial, cf_out, cf_eval)

        orig_impl = recorded.snapshots[target_idx].impl_name
        resolved = recorded.defect_category != "FAITHFUL" and cf_defect == "FAITHFUL"

        if resolved:
            verdict = (
                f"ROOT CAUSE ISOLATED AT [{target_stage.value}]: Swapping `{orig_impl}` -> "
                f"`{replacement_node.impl_name}` (reusing {len(reused_stage_names)} upstream stages) "
                f"eliminated `{recorded.defect_category}` and restored `FAITHFUL` output "
                f"(citations: {recorded.final_output.cited_doc_ids} -> {cf_out.cited_doc_ids})."
            )
        else:
            verdict = (
                f"INTERVENTION AT [{target_stage.value}] (`{orig_impl}` -> `{replacement_node.impl_name}`): "
                f"Outcome shifted `{recorded.defect_category}` -> `{cf_defect}`."
            )

        replayed_traj = RecordedTrajectory(
            trajectory_id=f"{recorded.trajectory_id}::CF[{target_stage.value}={replacement_node.impl_name}]",
            trial=recorded.trial,
            snapshots=new_snapshots,
            final_output=cf_out,
            evaluation=cf_eval,
            defect_category=cf_defect,
        )

        return CounterfactualReplayDiff(
            trajectory_id=recorded.trajectory_id,
            intervened_stage=target_stage,
            original_impl_name=orig_impl,
            replacement_impl_name=replacement_node.impl_name,
            reused_upstream_stages=reused_stage_names,
            reexecuted_downstream_stages=reexecuted_stage_names,
            original_defect_category=recorded.defect_category,
            counterfactual_defect_category=cf_defect,
            original_cited_docs=list(recorded.final_output.cited_doc_ids),
            counterfactual_cited_docs=list(cf_out.cited_doc_ids),
            original_abstained=recorded.final_output.abstained,
            counterfactual_abstained=cf_out.abstained,
            defect_resolved=resolved,
            causal_attribution_verdict=verdict,
            replayed_trajectory=replayed_traj,
        )

    def isolate_root_cause(
        self, recorded: RecordedTrajectory
    ) -> list[CounterfactualReplayDiff]:
        """
        Sweeps single-node counterfactual upgrades across:
          - Stage 3: `RERANK_FILTER` (`NaiveTopKFilter` -> `ConflictAwareRRF`)
          - Stage 4: `CONFLICT_CHECK` (`UncheckedConflictPassthrough` -> `SupersessionConflictResolver`)
          - Stage 5: `SYNTHESIZE_WITH_CITATIONS` (`PermissivePromptSynthesizer` -> `StrictCitationEntailmentSynthesizer`)
        Returns the ordered list of counterfactual diffs to pinpoint which stage intervention resolves the failure.
        """
        interventions = [
            make_conflict_aware_rrf_filter_node(),
            make_supersession_conflict_check_node(),
            make_strict_entailment_synthesizer_node(),
        ]
        return [
            self.replay_with_node_swap(recorded, node) for node in interventions
        ]
