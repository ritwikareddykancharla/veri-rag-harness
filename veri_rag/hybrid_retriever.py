"""
Hybrid Lexical (BM25Okapi) + Dense Vector (N-gram / Hashing Embedding Cosine)
Retriever with Reciprocal Rank Fusion (RRF), Semantic Chunking, Metadata Filtering,
Document Version Freshness Weighting, and Citation Span Extraction.

Zero external dependencies (Python 3.14 standard library only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import math
import re
from typing import Any, Iterable, Optional


STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "under",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
    }
)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text into normalized lowercase alphanumeric/hyphenated terms."""
    raw_tokens = [tok.lower() for tok in TOKEN_PATTERN.findall(text)]
    if not remove_stopwords:
        return raw_tokens
    return [tok for tok in raw_tokens if tok not in STOPWORDS and len(tok) > 1]


@dataclass(frozen=True)
class Document:
    """Enterprise knowledge base document with governance metadata."""

    doc_id: str
    title: str
    content: str
    domain: str
    region: str = "GLOBAL"
    effective_date: str = "2026-01-01"
    authority_tier: int = 1  # 1 = primary policy/amendment, 2 = SOP, 3 = legacy/superseded
    superseded_by: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """Semantically chunked passage from a Document with character offsets."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    start_char: int
    end_char: int
    domain: str
    region: str
    effective_date: str
    authority_tier: int
    superseded_by: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetadataFilter:
    """Declarative metadata filter applied prior to scoring."""

    region: Optional[str] = None
    domain: Optional[str] = None
    min_effective_date: Optional[str] = None
    max_authority_tier: Optional[int] = None
    exclude_superseded: bool = False
    allow_global_region_fallback: bool = True

    def matches(self, chunk: DocumentChunk) -> bool:
        if self.domain is not None and chunk.domain != self.domain:
            return False
        if self.region is not None:
            if chunk.region != self.region:
                if not (self.allow_global_region_fallback and chunk.region == "GLOBAL"):
                    return False
        if self.min_effective_date is not None:
            if chunk.effective_date < self.min_effective_date:
                return False
        if self.max_authority_tier is not None:
            if chunk.authority_tier > self.max_authority_tier:
                return False
        if self.exclude_superseded and chunk.superseded_by is not None:
            return False
        return True


@dataclass(frozen=True)
class CitationSpan:
    """Extracted verbatim sentence/clause span supporting a query."""

    doc_id: str
    chunk_id: str
    quote: str
    start_char: int
    end_char: int
    overlap_score: float


@dataclass(frozen=True)
class RetrievedResult:
    """Single ranked chunk produced by the HybridRRFRetriever."""

    chunk: DocumentChunk
    lexical_score: float
    vector_score: float
    rrf_score: float
    freshness_multiplier: float
    final_score: float
    lexical_rank: Optional[int]
    vector_rank: Optional[int]
    citation_span: CitationSpan


class SemanticChunker:
    """
    Splits enterprise policy and clinical protocol documents along section and sentence
    boundaries while preserving character offsets and governance metadata.
    """

    def __init__(self, max_chars: int = 420, overlap_chars: int = 60) -> None:
        if max_chars <= overlap_chars:
            raise ValueError("max_chars must be strictly greater than overlap_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_document(self, doc: Document) -> list[DocumentChunk]:
        text = doc.content
        if not text.strip():
            return []

        if len(text) <= self.max_chars:
            return [
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}#c0",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text.strip(),
                    start_char=0,
                    end_char=len(text),
                    domain=doc.domain,
                    region=doc.region,
                    effective_date=doc.effective_date,
                    authority_tier=doc.authority_tier,
                    superseded_by=doc.superseded_by,
                    metadata=dict(doc.metadata),
                )
            ]

        # Identify sentence spans with exact offsets
        sentence_spans: list[tuple[int, int]] = []
        cursor = 0
        for match in re.finditer(r"[^.!?\n]+[.!?\n]+|[^.!?\n]+$", text):
            s_start, s_end = match.span()
            if text[s_start:s_end].strip():
                sentence_spans.append((s_start, s_end))
            cursor = s_end
        if not sentence_spans:
            sentence_spans = [(0, len(text))]

        chunks: list[DocumentChunk] = []
        chunk_idx = 0
        i = 0
        n_sents = len(sentence_spans)

        while i < n_sents:
            start_char = sentence_spans[i][0]
            end_char = sentence_spans[i][1]
            j = i + 1
            while j < n_sents and (sentence_spans[j][1] - start_char) <= self.max_chars:
                end_char = sentence_spans[j][1]
                j += 1

            # If a single sentence exceeds max_chars, slice it cleanly at max_chars
            if end_char - start_char > self.max_chars and j == i + 1:
                sub_start = start_char
                while sub_start < end_char:
                    sub_end = min(sub_start + self.max_chars, end_char)
                    raw_slice = text[sub_start:sub_end].strip()
                    if raw_slice:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc.doc_id}#c{chunk_idx}",
                                doc_id=doc.doc_id,
                                title=doc.title,
                                text=raw_slice,
                                start_char=sub_start,
                                end_char=sub_end,
                                domain=doc.domain,
                                region=doc.region,
                                effective_date=doc.effective_date,
                                authority_tier=doc.authority_tier,
                                superseded_by=doc.superseded_by,
                                metadata=dict(doc.metadata),
                            )
                        )
                        chunk_idx += 1
                    if sub_end >= end_char:
                        break
                    sub_start = max(sub_start + 1, sub_end - self.overlap_chars)
                i = j
                continue

            raw_slice = text[start_char:end_char].strip()
            if raw_slice:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}#c{chunk_idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=raw_slice,
                        start_char=start_char,
                        end_char=end_char,
                        domain=doc.domain,
                        region=doc.region,
                        effective_date=doc.effective_date,
                        authority_tier=doc.authority_tier,
                        superseded_by=doc.superseded_by,
                        metadata=dict(doc.metadata),
                    )
                )
                chunk_idx += 1

            if j >= n_sents:
                break

            # Step back to preserve overlap_chars where possible
            next_i = j
            back_i = j - 1
            while back_i > i and (end_char - sentence_spans[back_i][0]) <= self.overlap_chars:
                next_i = back_i
                back_i -= 1
            i = max(i + 1, next_i)

        return chunks


class BM25OkapiIndex:
    """
    Pure-Python BM25Okapi lexical index over DocumentChunks.
    """

    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.num_docs = len(chunks)

        self.doc_term_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}

        for chunk in chunks:
            full_text = f"{chunk.title} {chunk.text}"
            tokens = tokenize(full_text, remove_stopwords=True)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_dl = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 1.0
        )
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(
                1.0 + (self.num_docs - df + 0.5) / (df + 0.5)
            )

    def score_subset(
        self, query: str, allowed_indices: Optional[set[int]] = None
    ) -> dict[int, float]:
        query_terms = tokenize(query, remove_stopwords=True)
        if not query_terms or self.num_docs == 0:
            return {}

        scores: dict[int, float] = {}
        indices: Iterable[int] = (
            allowed_indices if allowed_indices is not None else range(self.num_docs)
        )
        for idx in indices:
            tf_map = self.doc_term_freqs[idx]
            dl = self.doc_lengths[idx]
            norm_denom_base = self.k1 * (
                1.0 - self.b + self.b * (dl / max(self.avg_dl, 1e-9))
            )
            doc_score = 0.0
            for q_term in query_terms:
                tf = tf_map.get(q_term, 0)
                if tf == 0:
                    continue
                idf_val = self.idf.get(q_term, 0.0)
                doc_score += idf_val * ((tf * (self.k1 + 1.0)) / (tf + norm_denom_base))
            scores[idx] = doc_score
        return scores


class HashingVectorIndex:
    """
    Dense N-gram & word-shingle feature-hashing vector index (Weinberger et al.)
    with sublinear TF scaling and exact L2-normalized cosine similarity.
    Captures morphological variants, domain acronyms, and multi-word concepts.
    """

    def __init__(self, chunks: list[DocumentChunk], dim: int = 512) -> None:
        self.chunks = chunks
        self.dim = dim
        self.vectors: list[list[float]] = [
            self.embed_text(f"{c.title} {c.text}") for c in chunks
        ]

    @staticmethod
    def _extract_features(text: str) -> dict[str, float]:
        tokens = tokenize(text, remove_stopwords=True)
        features: dict[str, float] = {}

        # 1. Word unigrams (weight 1.0)
        for tok in tokens:
            key = f"u:{tok}"
            features[key] = features.get(key, 0.0) + 1.0
            # Stem-like 4-char and 5-char prefixes for morphological resilience
            if len(tok) >= 5:
                pref = f"p5:{tok[:5]}"
                features[pref] = features.get(pref, 0.0) + 0.65
            if len(tok) >= 4:
                pref4 = f"p4:{tok[:4]}"
                features[pref4] = features.get(pref4, 0.0) + 0.45

        # 2. Word bigrams (weight 1.25)
        for i in range(len(tokens) - 1):
            bg = f"b:{tokens[i]}_{tokens[i + 1]}"
            features[bg] = features.get(bg, 0.0) + 1.25

        # 3. Character 4-grams within content tokens (weight 0.35)
        for tok in tokens:
            if len(tok) >= 4:
                padded = f"^{tok}$"
                for i in range(len(padded) - 3):
                    cg = f"c4:{padded[i : i + 4]}"
                    features[cg] = features.get(cg, 0.0) + 0.35

        return features

    def embed_text(self, text: str) -> list[float]:
        features = self._extract_features(text)
        vec = [0.0] * self.dim
        for feat, raw_weight in features.items():
            # Deterministic MD5 hash for bucket index + sign bit
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            scaled = 1.0 + math.log(max(raw_weight, 1e-6))
            vec[bucket] += sign * scaled

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def score_subset(
        self, query: str, allowed_indices: Optional[set[int]] = None
    ) -> dict[int, float]:
        q_vec = self.embed_text(query)
        indices: Iterable[int] = (
            allowed_indices if allowed_indices is not None else range(len(self.chunks))
        )
        scores: dict[int, float] = {}
        for idx in indices:
            scores[idx] = max(0.0, self.cosine_similarity(q_vec, self.vectors[idx]))
        return scores


class HybridRRFRetriever:
    """
    Production Hybrid Retriever combining BM25Okapi lexical ranking and
    Dense N-gram Hashing Vector cosine ranking via Reciprocal Rank Fusion (RRF),
    augmented with metadata filtering, document version freshness weighting,
    and fine-grained citation span extraction.
    """

    def __init__(
        self,
        documents: list[Document],
        chunker: Optional[SemanticChunker] = None,
        vector_dim: int = 512,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.doc_map: dict[str, Document] = {d.doc_id: d for d in self.documents}
        self.chunker = chunker or SemanticChunker(max_chars=420, overlap_chars=60)

        self.chunks: list[DocumentChunk] = []
        for doc in self.documents:
            self.chunks.extend(self.chunker.chunk_document(doc))

        self.bm25_index = BM25OkapiIndex(self.chunks, k1=bm25_k1, b=bm25_b)
        self.vector_index = HashingVectorIndex(self.chunks, dim=vector_dim)

    @staticmethod
    def compute_freshness_multiplier(
        chunk: DocumentChunk,
        freshness_weight: float = 0.45,
        reference_date: date = date(2026, 11, 1),
    ) -> float:
        """
        Computes a multiplicative freshness & authority factor.
        - Newer effective dates receive a positive boost.
        - Authority tier 1 > tier 2 > tier 3.
        - Explicitly superseded documents receive a decay penalty when freshness_weight > 0.
        """
        if freshness_weight <= 0.0:
            return 1.0

        try:
            eff_date = date.fromisoformat(chunk.effective_date)
        except ValueError:
            eff_date = date(2025, 1, 1)

        days_old = max(0, (reference_date - eff_date).days)
        years_old = days_old / 365.25
        recency_factor = math.exp(-0.30 * years_old)

        tier_bonus = {1: 1.12, 2: 1.00, 3: 0.72}.get(chunk.authority_tier, 1.0)
        superseded_penalty = 0.48 if chunk.superseded_by is not None else 1.0

        raw_multiplier = recency_factor * tier_bonus * superseded_penalty
        return (1.0 - freshness_weight) + freshness_weight * (raw_multiplier * 1.25)

    @staticmethod
    def extract_citation_span(chunk: DocumentChunk, query: str) -> CitationSpan:
        """
        Extracts the most informative verbatim sentence/clause span from `chunk`
        to ground citations with character-level provenance.
        """
        query_terms = set(tokenize(query, remove_stopwords=True))
        text = chunk.text
        sentence_matches = list(re.finditer(r"[^.!?\n]+[.!?\n]*", text))
        if not sentence_matches:
            return CitationSpan(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                quote=text.strip(),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                overlap_score=0.0,
            )

        best_quote = text.strip()
        best_start = chunk.start_char
        best_end = chunk.end_char
        best_score = -1.0

        for match in sentence_matches:
            sent = match.group(0).strip()
            if not sent:
                continue
            sent_tokens = set(tokenize(sent, remove_stopwords=True))
            overlap = len(query_terms & sent_tokens) / max(len(query_terms), 1)
            # Reward sentences that carry numeric/policy constraints (e.g. %, hours, days, $, °C, mg)
            numeric_bonus = 0.15 if re.search(r"\d+", sent) else 0.0
            score = overlap + numeric_bonus
            if score > best_score:
                best_score = score
                best_quote = sent
                best_start = chunk.start_char + match.start()
                best_end = chunk.start_char + match.end()

        return CitationSpan(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            quote=best_quote,
            start_char=best_start,
            end_char=best_end,
            overlap_score=round(max(0.0, best_score), 4),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid_rrf",
        metadata_filter: Optional[MetadataFilter] = None,
        freshness_weight: float = 0.45,
        rrf_k: int = 20,
        lexical_weight: float = 0.55,
        vector_weight: float = 0.45,
    ) -> list[RetrievedResult]:
        """
        Retrieve top_k chunks using:
          - 'lexical_only': BM25Okapi ranking
          - 'vector_only': Hashing N-gram vector cosine ranking
          - 'hybrid_rrf': Reciprocal Rank Fusion + Freshness/Authority weighting
          - 'hybrid_rrf_no_freshness': RRF without freshness/authority multiplier
        """
        if mode not in {
            "lexical_only",
            "vector_only",
            "hybrid_rrf",
            "hybrid_rrf_no_freshness",
        }:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        allowed_indices: set[int] = set()
        for idx, chunk in enumerate(self.chunks):
            if metadata_filter is None or metadata_filter.matches(chunk):
                allowed_indices.add(idx)

        if not allowed_indices:
            return []

        bm25_scores = self.bm25_index.score_subset(query, allowed_indices)
        vec_scores = self.vector_index.score_subset(query, allowed_indices)

        # Compute deterministic ranks (1-indexed)
        lexical_sorted = sorted(
            allowed_indices,
            key=lambda i: (bm25_scores.get(i, 0.0), -self.chunks[i].authority_tier, self.chunks[i].effective_date),
            reverse=True,
        )
        vector_sorted = sorted(
            allowed_indices,
            key=lambda i: (vec_scores.get(i, 0.0), -self.chunks[i].authority_tier, self.chunks[i].effective_date),
            reverse=True,
        )

        lexical_rank_map = {idx: rank + 1 for rank, idx in enumerate(lexical_sorted)}
        vector_rank_map = {idx: rank + 1 for rank, idx in enumerate(vector_sorted)}

        effective_freshness_weight = (
            0.0 if mode in {"lexical_only", "vector_only", "hybrid_rrf_no_freshness"} else freshness_weight
        )

        results: list[RetrievedResult] = []
        for idx in allowed_indices:
            chunk = self.chunks[idx]
            lex_s = bm25_scores.get(idx, 0.0)
            vec_s = vec_scores.get(idx, 0.0)
            l_rank = lexical_rank_map[idx]
            v_rank = vector_rank_map[idx]

            rrf_s = (lexical_weight / (rrf_k + l_rank)) + (vector_weight / (rrf_k + v_rank))
            # Blend normalized raw signal with RRF rank score for sharp tie-breaking
            fresh_mult = self.compute_freshness_multiplier(
                chunk, freshness_weight=effective_freshness_weight
            )

            if mode == "lexical_only":
                final_s = lex_s
            elif mode == "vector_only":
                final_s = vec_s
            else:
                final_s = (rrf_s + 0.002 * lex_s + 0.01 * vec_s) * fresh_mult

            span = self.extract_citation_span(chunk, query)
            results.append(
                RetrievedResult(
                    chunk=chunk,
                    lexical_score=round(lex_s, 5),
                    vector_score=round(vec_s, 5),
                    rrf_score=round(rrf_s, 6),
                    freshness_multiplier=round(fresh_mult, 4),
                    final_score=round(final_s, 6),
                    lexical_rank=l_rank,
                    vector_rank=v_rank,
                    citation_span=span,
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)

        # Deduplicate by doc_id if multiple chunks from the same doc appear, keeping highest-scoring chunk
        deduped: list[RetrievedResult] = []
        seen_docs: set[str] = set()
        for res in results:
            if res.chunk.doc_id in seen_docs:
                continue
            seen_docs.add(res.chunk.doc_id)
            deduped.append(res)
            if len(deduped) >= top_k:
                break

        return deduped

    def evaluate_retrieval(
        self,
        scenarios: list[Any],
        top_k: int = 3,
        mode: str = "hybrid_rrf",
        use_region_filter: bool = True,
    ) -> dict[str, float]:
        """
        Evaluates a retrieval mode across answerable scenarios with ground-truth doc IDs:
          - recall_at_k
          - top1_hit_rate
          - mrr_at_k
          - ndcg_at_k
          - fresh_conflict_win_rate (how often the 2026 policy outranks its 2024 superseded doc)
        """
        answerable = [s for s in scenarios if not s.should_abstain and s.expected_doc_ids]
        if not answerable:
            return {
                "recall_at_k": 0.0,
                "top1_hit_rate": 0.0,
                "mrr_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "fresh_conflict_win_rate": 0.0,
            }

        recall_sum = 0.0
        top1_hits = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        conflict_trials = 0
        conflict_wins = 0

        for scen in answerable:
            m_filter = (
                MetadataFilter(region=scen.region, allow_global_region_fallback=True)
                if use_region_filter
                else None
            )
            retrieved = self.retrieve(
                scen.query,
                top_k=max(top_k, 5),
                mode=mode,
                metadata_filter=m_filter,
            )
            top_k_docs = [r.chunk.doc_id for r in retrieved[:top_k]]
            expected_set = set(scen.expected_doc_ids)

            hits = len(expected_set & set(top_k_docs))
            recall_sum += hits / len(expected_set)

            if top_k_docs and top_k_docs[0] in expected_set:
                top1_hits += 1

            rr = 0.0
            for rank_idx, doc_id in enumerate(top_k_docs):
                if doc_id in expected_set:
                    rr = 1.0 / (rank_idx + 1)
                    break
            mrr_sum += rr

            dcg = 0.0
            for rank_idx, doc_id in enumerate(top_k_docs):
                if doc_id in expected_set:
                    dcg += 1.0 / math.log2(rank_idx + 2)
            ideal_hits = min(len(expected_set), top_k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

            if getattr(scen, "stale_conflicting_doc_id", None):
                conflict_trials += 1
                all_ranked_docs = [r.chunk.doc_id for r in retrieved]
                fresh_doc = scen.expected_doc_ids[0]
                stale_doc = scen.stale_conflicting_doc_id
                fresh_pos = (
                    all_ranked_docs.index(fresh_doc)
                    if fresh_doc in all_ranked_docs
                    else 999
                )
                stale_pos = (
                    all_ranked_docs.index(stale_doc)
                    if stale_doc in all_ranked_docs
                    else 999
                )
                if fresh_pos < stale_pos:
                    conflict_wins += 1

        n = len(answerable)
        return {
            "recall_at_k": round(recall_sum / n, 4),
            "top1_hit_rate": round(top1_hits / n, 4),
            "mrr_at_k": round(mrr_sum / n, 4),
            "ndcg_at_k": round(ndcg_sum / n, 4),
            "fresh_conflict_win_rate": round(
                conflict_wins / max(conflict_trials, 1), 4
            ),
        }
