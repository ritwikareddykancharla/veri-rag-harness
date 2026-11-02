"""
Dataset Contamination Checker (N-gram Jaccard & Hashing Embedding Cosine Scanner).

Scans golden evaluation questions and reference answers against retrieval corpora
and post-training / fine-tuning splits to detect verbatim benchmark leakage,
paraphrase contamination, and synthetic Q&A overlap before model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from veri_rag.hybrid_retriever import Document, HashingVectorIndex, tokenize
from veri_rag.scenarios import EvaluationScenario


def extract_word_ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    """Extracts contiguous word n-gram tuples from normalized text."""
    tokens = tokenize(text, remove_stopwords=False)
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_jaccard_similarity(text_a: str, text_b: str, n: int = 4) -> float:
    """Computes word n-gram Jaccard similarity |A ∩ B| / |A ∪ B|."""
    ngrams_a = extract_word_ngrams(text_a, n=n)
    ngrams_b = extract_word_ngrams(text_b, n=n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    union = ngrams_a | ngrams_b
    if not union:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(union)


def ngram_containment_score(query_text: str, doc_text: str, n: int = 5) -> float:
    """Computes fraction of query 5-grams that appear verbatim inside `doc_text`."""
    q_ngrams = extract_word_ngrams(query_text, n=n)
    d_ngrams = extract_word_ngrams(doc_text, n=n)
    if not q_ngrams or not d_ngrams:
        return 0.0
    return len(q_ngrams & d_ngrams) / len(q_ngrams)


@dataclass(frozen=True)
class ScenarioContaminationResult:
    """Contamination scan verdict for a single golden evaluation scenario."""

    scenario_id: str
    status: str  # "CLEAN", "BORDERLINE_OVERLAP", "CONTAMINATED_LEAK"
    closest_doc_id: str
    max_ngram_jaccard_4: float
    max_query_5gram_containment: float
    max_embedding_cosine: float
    Composite_leak_score: float
    leak_reason: Optional[str]


@dataclass(frozen=True)
class ContaminationScanReport:
    """Aggregate report across all scanned evaluation scenarios."""

    total_scenarios_scanned: int
    total_corpus_docs_scanned: int
    clean_count: int
    borderline_count: int
    contaminated_leak_count: int
    contamination_rate: float
    flagged_scenario_ids: list[str]
    results: list[ScenarioContaminationResult]


class DatasetContaminationChecker:
    """
    Scans golden evaluation benchmarks against a retrieval corpus or SFT training
    split using 4-gram Jaccard, 5-gram query containment, and 512-d hashing cosine.
    """

    def __init__(
        self,
        jaccard_leak_threshold: float = 0.28,
        query_containment_leak_threshold: float = 0.50,
        cosine_leak_threshold: float = 0.86,
    ) -> None:
        self.jaccard_leak_threshold = jaccard_leak_threshold
        self.query_containment_leak_threshold = query_containment_leak_threshold
        self.cosine_leak_threshold = cosine_leak_threshold
        self._vectorizer = HashingVectorIndex([], dim=512)

    def scan(
        self,
        scenarios: list[EvaluationScenario],
        corpus_or_training_docs: list[Document],
    ) -> ContaminationScanReport:
        doc_vectors = {
            doc.doc_id: self._vectorizer.embed_text(f"{doc.title} {doc.content}")
            for doc in corpus_or_training_docs
        }

        results: list[ScenarioContaminationResult] = []
        clean_count = 0
        borderline_count = 0
        leak_count = 0
        flagged_ids: list[str] = []

        for scen in scenarios:
            eval_probe = f"{scen.query} {scen.ground_truth_answer}"
            q_vec = self._vectorizer.embed_text(eval_probe)

            best_doc_id = "NONE"
            max_jaccard = 0.0
            max_q_containment = 0.0
            max_cosine = 0.0
            best_composite = -1.0

            for doc in corpus_or_training_docs:
                full_doc_text = f"{doc.title} {doc.content}"
                jac = ngram_jaccard_similarity(eval_probe, full_doc_text, n=4)
                q_cont = ngram_containment_score(scen.query, full_doc_text, n=5)
                cos = max(
                    0.0,
                    HashingVectorIndex.cosine_similarity(q_vec, doc_vectors[doc.doc_id]),
                )
                composite = 0.45 * q_cont + 0.35 * jac + 0.20 * cos

                if composite > best_composite:
                    best_composite = composite
                    best_doc_id = doc.doc_id
                    max_jaccard = jac
                    max_q_containment = q_cont
                    max_cosine = cos

            # Classify contamination status
            is_leak = (
                max_q_containment >= self.query_containment_leak_threshold
                or max_jaccard >= self.jaccard_leak_threshold
                or max_cosine >= self.cosine_leak_threshold
            )
            is_borderline = (
                not is_leak
                and (
                    max_q_containment >= 0.30
                    or max_jaccard >= 0.18
                    or max_cosine >= 0.72
                )
            )

            if is_leak:
                status = "CONTAMINATED_LEAK"
                leak_count += 1
                flagged_ids.append(scen.scenario_id)
                reason = (
                    f"Verbatim 5-gram query containment={max_q_containment:.2f} / "
                    f"4-gram Jaccard={max_jaccard:.2f} in [{best_doc_id}]"
                )
            elif is_borderline:
                status = "BORDERLINE_OVERLAP"
                borderline_count += 1
                reason = f"Moderate n-gram/semantic overlap with [{best_doc_id}]"
            else:
                status = "CLEAN"
                clean_count += 1
                reason = None

            results.append(
                ScenarioContaminationResult(
                    scenario_id=scen.scenario_id,
                    status=status,
                    closest_doc_id=best_doc_id,
                    max_ngram_jaccard_4=round(max_jaccard, 4),
                    max_query_5gram_containment=round(max_q_containment, 4),
                    max_embedding_cosine=round(max_cosine, 4),
                    Composite_leak_score=round(max(0.0, best_composite), 4),
                    leak_reason=reason,
                )
            )

        n_scen = max(len(scenarios), 1)
        return ContaminationScanReport(
            total_scenarios_scanned=len(scenarios),
            total_corpus_docs_scanned=len(corpus_or_training_docs),
            clean_count=clean_count,
            borderline_count=borderline_count,
            contaminated_leak_count=leak_count,
            contamination_rate=round(leak_count / n_scen, 4),
            flagged_scenario_ids=flagged_ids,
            results=results,
        )
