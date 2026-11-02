"""
VeriRAG: Hybrid Retrieval Engine, Adversarial Abstention Harness,
5% Shadow Traffic Router & Counterfactual Replay Debugger.
"""

from veri_rag.adversarial_harness import (
    AdversarialAbstentionHarness,
    AdversarialStressMode,
    GenerationOutput,
    HarnessMetricsSummary,
    NaiveRAGGenerator,
    PerturbedTrial,
    TrialEvaluationResult,
    VeriRAGGenerator,
)
from veri_rag.dataset_contamination import (
    ContaminationScanReport,
    DatasetContaminationChecker,
    ScenarioContaminationResult,
)
from veri_rag.hybrid_retriever import (
    BM25OkapiIndex,
    CitationSpan,
    Document,
    DocumentChunk,
    HashingVectorIndex,
    HybridRRFRetriever,
    MetadataFilter,
    RetrievedResult,
    SemanticChunker,
)
from veri_rag.replay_debugger import (
    CounterfactualReplayDebugger,
    CounterfactualReplayDiff,
    NodeExecutionSnapshot,
    RecordedTrajectory,
    WorkflowNode,
    WorkflowStage,
)
from veri_rag.scenarios import (
    EvaluationScenario,
    build_enterprise_corpus,
    build_golden_scenarios,
)
from veri_rag.shadow_router import (
    BootstrapMetricCI,
    ShadowComparisonRecord,
    ShadowRouterReport,
    ShadowTrafficRouter,
)

__version__ = "1.0.0"

__all__ = [
    "AdversarialAbstentionHarness",
    "AdversarialStressMode",
    "BM25OkapiIndex",
    "BootstrapMetricCI",
    "CitationSpan",
    "ContaminationScanReport",
    "CounterfactualReplayDebugger",
    "CounterfactualReplayDiff",
    "DatasetContaminationChecker",
    "Document",
    "DocumentChunk",
    "EvaluationScenario",
    "GenerationOutput",
    "HarnessMetricsSummary",
    "HashingVectorIndex",
    "HybridRRFRetriever",
    "MetadataFilter",
    "NaiveRAGGenerator",
    "NodeExecutionSnapshot",
    "PerturbedTrial",
    "RecordedTrajectory",
    "RetrievedResult",
    "ScenarioContaminationResult",
    "SemanticChunker",
    "ShadowComparisonRecord",
    "ShadowRouterReport",
    "ShadowTrafficRouter",
    "TrialEvaluationResult",
    "VeriRAGGenerator",
    "WorkflowNode",
    "WorkflowStage",
    "build_enterprise_corpus",
    "build_golden_scenarios",
]
