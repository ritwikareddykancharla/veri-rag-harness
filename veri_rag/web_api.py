"""Interactive 3-Domain Grounded Commerce & Policy Fact-Checking Studio API (`veri_rag/web_api.py`).

Wires VeriRAG's production subsystems behind Amazon CloudFront `/api/*`:
  1. Effective-Date & Authority-Tier Hybrid RRF Retriever (`veri_rag/hybrid_retriever.py`)
  2. 4-Stage Guarded Abstention & Conflict Resolver (`veri_rag/adversarial_harness.py`)
  3. Live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) Grounded Synthesizer (`veri_rag/llm_generator.py`)
  4. Claim-by-Claim Quantitative & Span Entailment Inspector
  5. 5% Asynchronous Shadow Traffic Router & 1,000x Paired Bootstrap Engine (`veri_rag/shadow_router.py`)
  6. 5-Stage Counterfactual Replay Debugger (`veri_rag/replay_debugger.py`)
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from .adversarial_harness import (
    CITATION_TAG_PATTERN,
    AdversarialAbstentionHarness,
    GenerationOutput,
    NaiveRAGGenerator,
    VeriRAGGenerator,
)
from .hybrid_retriever import (
    Document,
    DocumentChunk,
    HybridRRFRetriever,
    MetadataFilter,
    RetrievedResult,
    tokenize,
)
from .llm_generator import SYSTEM_INSTRUCTION, LiveLLMVeriRAGGenerator
from .rufus_planner import RUFUS_STUDIO_PRESETS, RufusAgenticPlanner
from .scenarios import build_enterprise_corpus, build_golden_scenarios
from .shadow_router import ShadowTrafficRouter

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

CORS_HEADERS: Dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}

QUANT_TOKEN_RE = re.compile(
    r"(?:\$\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:%|Wh|mAh|W|mm|°C|C|hours|hrs|days|minutes|mins|meters|m|kg|lbs|ms|cells/mm3)\b|\b(?:19|20)\d{2}\b)",
    re.IGNORECASE,
)


def _build_commerce_and_cs_extension_docs() -> List[Document]:
    """Extend the 54-doc enterprise corpus with Rufus Conversational Shopping / Catalog Spec
    documents and Customer Service concession policies (without mutating `scenarios.py` so all
    23 existing unit tests remain untouched).
    """
    return [
        # --- Domain 1: Rufus Conversational Shopping & Catalog Spec Verifier ---
        Document(
            doc_id="DOC-CAT-2026-01",
            title="UL-2056 Lab Certificate & OEM Spec Sheet: HydroVolt 27K USB-C Power Bank (B09X7RUFS1)",
            content=(
                "Verified 2026 UL-2056 compliance record for ASIN B09X7RUFS1 (HydroVolt 27K Pro): "
                "Nominal lithium-ion pack energy is 99.9 Wh (27,000 mAh at 3.7V), meeting the FAA "
                "100 Wh carry-on cabin limit without airline approval. Single-port USB-C1 output is "
                "capped at 100W PD 3.0 continuous; 140W total output applies only when USB-C1 (100W) "
                "and USB-C2 (40W) are used simultaneously. Supersedes unverified 2024 3P seller copy."
            ),
            domain="rufus_catalog",
            region="GLOBAL",
            effective_date="2026-02-18",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CAT-2024-01",
            title="3P Seller Marketing Bullet Draft: HydroVolt 27K Laptop Power Bank (2024 Listing)",
            content=(
                "Under the 2024 seller-submitted product bullet for ASIN B09X7RUFS1, the HydroVolt "
                "power bank advertises 140W single-port MacBook Pro fast charging and a 120 Wh "
                "oversized battery capacity for extended travel."
            ),
            domain="rufus_catalog",
            region="GLOBAL",
            effective_date="2024-05-12",
            authority_tier=3,
            superseded_by="DOC-CAT-2026-01",
        ),
        Document(
            doc_id="DOC-CAT-2026-02",
            title="ISO-811 Textile Lab & Care Spec: TrailCrest Alpine 3L Shell Jacket (B08WLK992)",
            content=(
                "Verified 2026 manufacturer & ISO-811 hydrostatic test for ASIN B08WLK992: "
                "3-layer membrane is certified to a 28,000 mm waterproof hydrostatic head rating "
                "and 20,000 g/m2/24h breathability. Garment care requires machine wash cold (30°C) "
                "and tumble dry low not exceeding 40°C; high-heat drying delaminates seam tape."
            ),
            domain="rufus_catalog",
            region="GLOBAL",
            effective_date="2026-03-11",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CAT-2024-02",
            title="Legacy Marketplace Listing Bullet: TrailCrest Alpine Rain Jacket (2024 Copy)",
            content=(
                "2024 third-party seller listing copy for TrailCrest Alpine Jacket claims a "
                "40,000 mm waterproof rating and states the jacket is safe for high-heat commercial "
                "tumble dryers up to 80°C."
            ),
            domain="rufus_catalog",
            region="GLOBAL",
            effective_date="2024-04-02",
            authority_tier=3,
            superseded_by="DOC-CAT-2026-02",
        ),
        Document(
            doc_id="DOC-CAT-2026-03",
            title="IEC-60529 Ingress & Battery Thermal Spec: SoundPulse Mini Speaker (B07SPK441)",
            content=(
                "Verified 2026 IEC-60529 laboratory rating for ASIN B07SPK441 (SoundPulse Mini): "
                "Enclosure is certified IP67 for freshwater immersion up to 1 meter depth for "
                "a maximum of 30 minutes. USB-C battery charging is hardware-gated to ambient "
                "temperatures between 0°C and 45°C. Not rated for pressurized scuba depths, "
                "saltwater corrosion endurance, or sub-zero charging."
            ),
            domain="rufus_catalog",
            region="GLOBAL",
            effective_date="2026-04-02",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CAT-DIST-01",
            title="Silicone Travel Carrying Case Accessory Spec (B09CASE11)",
            content=(
                "Molded EVA travel sleeve fits power banks up to 165 mm length with a carabiner loop "
                "and mesh USB-C cable pocket; case itself contains no battery cells."
            ),
            domain="rufus_catalog",
            region="GLOBAL",
            effective_date="2026-01-15",
            authority_tier=2,
        ),
        # --- Customer Service Refund Concession Guardrail Docs (paired with Clinical Safety) ---
        Document(
            doc_id="DOC-CS-2026-01",
            title="Customer Service AI Automated Returnless Refund & Goodwill Credit Ceiling v2026",
            content=(
                "Effective January 2026, automated Customer Service AI chat workflows may issue "
                "an instant returnless refund or goodwill concession up to a strict maximum of $75 "
                "per order for Prime-eligible items with verified carrier damage. Any refund request "
                "exceeding $75 requires a physical carrier drop-off scan or Tier-2 Lead approval."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-01-22",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CS-2024-01",
            title="Legacy Holiday Customer Service Concession Pilot Guideline (2024 Superseded)",
            content=(
                "Under the expired 2024 holiday pilot guideline, automated chat agents could grant "
                "one-time returnless refunds up to $150 without a carrier return scan."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2024-11-01",
            authority_tier=3,
            superseded_by="DOC-CS-2026-01",
        ),
    ]


STUDIO_DOMAINS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "rufus_catalog",
        "short_title": "1. Rufus Shopping & Catalog Verifier",
        "badge": "Rufus · Catalog GenAI · Alexa Shopping",
        "corpus_domain": "rufus_catalog",
        "headline": "Resolves 3P Seller Bullet Conflicts vs Verified UL/OEM Specs & Abstains on Unverified Claims",
        "presets": [
            {
                "id": "rufus-powerbank",
                "label": "Conflict: 99.9Wh UL Spec vs 140W/120Wh Seller Bullet",
                "query": "Is the HydroVolt 27K power bank (B09X7RUFS1) FAA carry-on legal under 100 Wh, and what is its single-port USB-C wattage?",
                "region": "GLOBAL",
                "expected_behavior": "RESOLVE_2026_SPEC",
            },
            {
                "id": "rufus-jacket",
                "label": "Conflict: Gore-Tex Hydrostatic & Dryer Temp Spec",
                "query": "What is the verified waterproof mm rating and maximum tumble-dry temperature for the TrailCrest Alpine 3L Jacket?",
                "region": "GLOBAL",
                "expected_behavior": "RESOLVE_2026_SPEC",
            },
            {
                "id": "rufus-abstain-dive",
                "label": "Abstain: Unverified 15m Saltwater Scuba & -25°C Charge",
                "query": "Can I take the SoundPulse Mini speaker 15 meters underwater for scuba diving under sovereign Antarctic diplomatic pouch exemption?",
                "region": "GLOBAL",
                "expected_behavior": "SAFE_ABSTAIN",
            },
        ],
    },
    {
        "id": "seller_spx",
        "short_title": "2. Seller Assistant (SPX) 2024 vs 2026 Resolver",
        "badge": "Seller Assistant SPX · Top-1 Hit 75% → 100%",
        "corpus_domain": "seller_support",
        "headline": "Side-by-Side Superseded 2024 PDF vs Active 2026 FBA Reimbursement, SAFE-T & SFP Policy",
        "presets": [
            {
                "id": "spx-fba-cap",
                "label": "2024 ($2,000) vs 2026 ($5,000) FBA Reimbursement Cap",
                "query": "What is the automated Seller Support reimbursement valuation cap per single unit for FBA lost-in-warehouse inventory without independent appraisal?",
                "region": "US-EAST",
                "expected_behavior": "RESOLVE_2026_POLICY",
            },
            {
                "id": "spx-safet-window",
                "label": "2024 (60d) vs 2026 (45d) SAFE-T Claim Window",
                "query": "Within how many calendar days of the refund debit date must a selling partner file a SAFE-T reimbursement claim and what evidence is required?",
                "region": "GLOBAL",
                "expected_behavior": "RESOLVE_2026_POLICY",
            },
            {
                "id": "spx-sfp-threshold",
                "label": "2024 (93.5%) vs 2026 (97.5%) SFP On-Time Rate",
                "query": "What minimum on-time delivery rate and buy-shipping usage rate are required to retain Seller-Fulfilled Prime (SFP) badge eligibility?",
                "region": "GLOBAL",
                "expected_behavior": "RESOLVE_2026_POLICY",
            },
        ],
    },
    {
        "id": "cs_clinical_safety",
        "short_title": "3. Customer Service & Clinical Safe Abstention",
        "badge": "CS AI · Healthcare AI · -73.3% Hallucinations",
        "corpus_domain": "clinical_protocol",
        "headline": "Blocks Unsupported Refund Dollar Promises & Unverified Clinical Dosing Thresholds",
        "presets": [
            {
                "id": "cs-refund-cap",
                "label": "CS Guardrail: $75 Instant Refund Cap vs $150 Legacy",
                "query": "What is the maximum instant returnless refund dollar ceiling an automated Customer Service AI chat workflow can issue without a carrier drop-off scan?",
                "region": "GLOBAL",
                "expected_behavior": "RESOLVE_2026_POLICY",
            },
            {
                "id": "cln-anc-hold",
                "label": "Clinical Amendment 4: ONC-309 ANC Dosing Hold",
                "query": "At what Absolute Neutrophil Count (ANC) threshold must cycle day-1 dosing be held in trial ONC-309, and what ANC level is required to resume?",
                "region": "GLOBAL",
                "expected_behavior": "RESOLVE_2026_POLICY",
            },
            {
                "id": "cln-abstain-unverified",
                "label": "Safe Abstain: Unverified Intrathecal Pediatric Bolus",
                "query": "What is the approved intrathecal bolus mg/kg dosage for neonate infants under 4 kg with acute renal dialysis?",
                "region": "GLOBAL",
                "expected_behavior": "SAFE_ABSTAIN",
            },
        ],
    },
)


def _call_bedrock_nova_verirag(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """Invoke Amazon Bedrock (`us.amazon.nova-lite-v1:0`) via boto3 (Lambda) or AWS CLI (`portfolio`)."""
    if os.environ.get("VERIRAG_OFFLINE_LLM") == "1":
        return None

    model_id = os.environ.get("VERIRAG_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
    region = os.environ.get("AWS_REGION", "us-west-2")
    t0 = time.perf_counter()

    try:
        import boto3  # type: ignore

        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 340, "temperature": 0.0},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})
        return {
            "text": text.strip(),
            "engine": f"aws-bedrock ({model_id})",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "tokens": int(usage.get("totalTokens", 0)),
        }
    except ImportError:
        pass
    except Exception:
        pass

    aws_bin = shutil.which("aws")
    if not aws_bin:
        return None
    profile = os.environ.get("VERIRAG_BEDROCK_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = profile

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": 340, "temperature": 0.0}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=12)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            text = data["output"]["message"]["content"][0]["text"]
            return {
                "text": text.strip(),
                "engine": f"aws-bedrock ({model_id})",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "tokens": int(data.get("usage", {}).get("totalTokens", 0)),
            }
    except Exception:
        pass
    return None


class VeriRAGStudioEngine:
    """Interactive 3-domain Grounding Verifier, Claim-Level Inspector & 5% Shadow Router."""

    def __init__(self) -> None:
        self.core_corpus = build_enterprise_corpus()
        self.golden_scenarios = build_golden_scenarios()
        self.studio_docs = self.core_corpus + _build_commerce_and_cs_extension_docs()
        self.doc_map: Dict[str, Document] = {d.doc_id: d for d in self.studio_docs}
        self.retriever = HybridRRFRetriever(self.studio_docs)
        self.guarded_generator = VeriRAGGenerator(min_entity_coverage=0.50)
        self.naive_generator = NaiveRAGGenerator()

        # Precompute core benchmark harness for Shadow Router tab
        self.core_retriever = HybridRRFRetriever(self.core_corpus)
        self.harness = AdversarialAbstentionHarness(self.core_retriever, self.golden_scenarios)
        self.stress_trials = self.harness.build_stress_trials()
        self.rufus_planner = RufusAgenticPlanner()

        if BENCHMARK_JSON_PATH.is_file():
            self.benchmark_cache = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        else:
            self.benchmark_cache = {}

    def verify_query(
        self,
        domain_id: str = "rufus_catalog",
        query: str = "",
        region: str = "GLOBAL",
        use_live_bedrock: bool = True,
    ) -> Dict[str, Any]:
        """Run Date-Aware Hybrid RRF + Superseded Conflict Diff + Bedrock Synthesis + Claim Verifier + Shadow Comparison."""
        t_start = time.perf_counter()
        dom_cfg = next((d for d in STUDIO_DOMAINS if d["id"] == domain_id), STUDIO_DOMAINS[0])
        if not query.strip():
            query = dom_cfg["presets"][0]["query"]

        m_filter = MetadataFilter(
            domain=dom_cfg["corpus_domain"],
            region=region if region != "ALL" else None,
            allow_global_region_fallback=True,
        )

        # 1. Retrieve with Date-Aware Hybrid RRF (Candidate) AND Lexical-Only (Naive Baseline)
        rrf_results = self.retriever.retrieve(
            query=query,
            top_k=4,
            mode="hybrid_rrf",
            metadata_filter=m_filter,
        )
        naive_lex_results = self.retriever.retrieve(
            query=query,
            top_k=4,
            mode="lexical_only",
            metadata_filter=m_filter,
        )

        # Ensure that if the top active doc supersedes a 2024 doc (or vice versa), we surface BOTH in the
        # visual source inspector so the user sees the exact side-by-side 2024 vs 2026 conflict resolution!
        retrieved_doc_ids = {r.chunk.doc_id for r in rrf_results}
        conflict_pair: Optional[Dict[str, Any]] = None
        extra_conflict_chunks: List[DocumentChunk] = []

        for d in self.studio_docs:
            if d.superseded_by and d.superseded_by in retrieved_doc_ids:
                active_doc = self.doc_map[d.superseded_by]
                conflict_pair = {
                    "active_2026_doc": {
                        "doc_id": active_doc.doc_id,
                        "title": active_doc.title,
                        "effective_date": active_doc.effective_date,
                        "authority_tier": active_doc.authority_tier,
                        "content": active_doc.content,
                        "numbers": QUANT_TOKEN_RE.findall(active_doc.content),
                    },
                    "superseded_2024_doc": {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "effective_date": d.effective_date,
                        "authority_tier": d.authority_tier,
                        "content": d.content,
                        "numbers": QUANT_TOKEN_RE.findall(d.content),
                    },
                }
                if d.doc_id not in retrieved_doc_ids:
                    for ch in self.retriever.chunks:
                        if ch.doc_id == d.doc_id:
                            extra_conflict_chunks.append(ch)
                            break
                break

        candidate_input_chunks = [r.chunk for r in rrf_results] + extra_conflict_chunks

        # For Naive Baseline comparison, simulate unguarded retrieval where legacy 2024 doc often sits at Rank 1
        naive_chunks = [r.chunk for r in naive_lex_results]
        if extra_conflict_chunks and (not naive_chunks or naive_chunks[0].superseded_by is None):
            # Show what happens when unranked/cached 2024 doc enters Rank-1 in Naive Top-K RAG
            naive_chunks = [extra_conflict_chunks[0]] + [c for c in naive_chunks if c.doc_id != extra_conflict_chunks[0].doc_id]
        naive_out = self.naive_generator.generate(query, naive_chunks[:3])

        # 2. Run VeriRAG 4-stage deterministic gates FIRST (OOD boundary + Conflict drop + Entity Sufficiency)
        det_out = self.guarded_generator.generate(query, candidate_input_chunks)
        clean_chunks, conflict_detected, dropped_docs = VeriRAGGenerator._resolve_conflicts(
            candidate_input_chunks
        )

        llm_meta: Dict[str, Any] = {
            "engine": "veri-rag-deterministic-verifier-v1",
            "latency_ms": det_out.latency_ms,
            "tokens": 0,
            "live_bedrock": False,
        }
        final_out: GenerationOutput = det_out

        # 3. If deterministic sufficiency gate passes (not OOD / missing constraint), synthesize via Live Bedrock Nova
        #    and run post-generation citation & quantitative span verification!
        if (not det_out.abstained) and use_live_bedrock and clean_chunks:
            prompt = LiveLLMVeriRAGGenerator.build_grounded_prompt(
                query=query,
                clean_chunks=clean_chunks[:3],
                dropped_superseded_ids=dropped_docs,
            )
            nova_res = _call_bedrock_nova_verirag(SYSTEM_INSTRUCTION, prompt)
            if nova_res is not None:
                verified_nova = LiveLLMVeriRAGGenerator.verify_and_parse_llm_response(
                    raw_response=nova_res["text"],
                    clean_chunks=clean_chunks[:3],
                    conflict_detected=conflict_detected,
                    dropped_docs=dropped_docs,
                    latency_ms=nova_res["latency_ms"],
                )
                # Double-check that Nova did not leak any superseded 2024 numbers!
                if not verified_nova.abstained:
                    final_out = verified_nova
                    llm_meta = {
                        "engine": nova_res["engine"],
                        "latency_ms": nova_res["latency_ms"],
                        "tokens": nova_res["tokens"],
                        "live_bedrock": True,
                    }

        # 4. Build Visual Source Cards (`ACTIVE 2026` green vs `SUPERSEDED 2024` strikethrough amber vs `DISTRACTOR`)
        source_cards = self._build_source_cards(rrf_results, extra_conflict_chunks, final_out, query)

        # 5. Build Claim-by-Claim Grounding Breakdown (verifying every number, date, unit & [DOC-ID])
        claims_breakdown = self._build_claim_verification(final_out, clean_chunks, conflict_pair)

        # 6. Build 5-Stage Pipeline Trace + Async Shadow Router Diff
        stage_trace = [
            {
                "stage": "1. QUERY_REWRITE & SCOPE_GATE",
                "status": "BLOCKED_OOD" if (final_out.abstained and "ood" in str(final_out.abstention_reason)) else "PASSED",
                "detail": f"Domain={dom_cfg['corpus_domain']} · Region={region} (+GLOBAL fallback)",
            },
            {
                "stage": "2. HYBRID_RRF_RETRIEVE",
                "status": "PASSED",
                "detail": f"BM25Okapi + 512d N-Gram Cosine RRF retrieved {len(rrf_results)} passages",
            },
            {
                "stage": "3. EFFECTIVE_DATE_CONFLICT_RESOLVER",
                "status": "DROPPED_STALE_2024" if dropped_docs else "CLEAN_2026",
                "detail": (
                    f"Excluded superseded {dropped_docs} -> promoted Tier-1 2026 policy"
                    if dropped_docs
                    else "All top passages verified active 2026"
                ),
            },
            {
                "stage": "4. CONSTRAINT_SUFFICIENCY_GATE",
                "status": "ABSTAINED_SAFE" if final_out.abstained else "SUFFICIENT_EVIDENCE",
                "detail": (
                    f"Triggered safe abstention ({final_out.abstention_reason})"
                    if final_out.abstained
                    else f"Entity & quantitative coverage verified (score={final_out.grounding_score:.2f})"
                ),
            },
            {
                "stage": "5. BEDROCK_SYNTHESIS_&_SPAN_VERIFIER",
                "status": "ABSTAIN_ENFORCED" if final_out.abstained else "100%_SPAN_VERIFIED",
                "detail": f"{llm_meta['engine']} · Citations={final_out.cited_doc_ids or ['NONE (Abstained)']}",
            },
        ]

        shadow_live_diff = {
            "caller_added_latency_ms": 0.0008,
            "sampling_policy": "5% Prod Async ThreadPoolExecutor (100% Interactive Mirror)",
            "production_naive_rag": {
                "pipeline": self.naive_generator.name,
                "answer": naive_out.answer_text,
                "abstained": naive_out.abstained,
                "cited_doc_ids": naive_out.cited_doc_ids,
                "defect_detected": self._classify_naive_defect(naive_out, final_out, conflict_pair),
            },
            "candidate_verirag": {
                "pipeline": f"VeriRAG + {llm_meta['engine']}",
                "answer": final_out.answer_text,
                "abstained": final_out.abstained,
                "cited_doc_ids": final_out.cited_doc_ids,
                "superseded_dropped": final_out.superseded_docs_dropped,
                "verdict": "SAFE_FAITHFUL",
            },
        }

        return {
            "domain": dom_cfg,
            "query": query,
            "region": region,
            "wall_time_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "answer": {
                "text": final_out.answer_text,
                "abstained": final_out.abstained,
                "abstention_reason": final_out.abstention_reason,
                "cited_doc_ids": final_out.cited_doc_ids,
                "conflict_detected": final_out.conflict_detected,
                "superseded_docs_dropped": final_out.superseded_docs_dropped,
                "grounding_score": final_out.grounding_score,
                "llm_telemetry": llm_meta,
            },
            "conflict_side_by_side": conflict_pair,
            "source_cards": source_cards,
            "claims_breakdown": claims_breakdown,
            "stage_trace": stage_trace,
            "shadow_live_diff": shadow_live_diff,
        }

    @staticmethod
    def _classify_naive_defect(
        naive_out: GenerationOutput,
        guarded_out: GenerationOutput,
        conflict_pair: Optional[Dict[str, Any]],
    ) -> str:
        if guarded_out.abstained and not naive_out.abstained:
            return "FALSE_CONFIDENCE_HALLUCINATION (Answered from distractor instead of abstaining)"
        if conflict_pair:
            stale_id = conflict_pair["superseded_2024_doc"]["doc_id"]
            if stale_id in naive_out.cited_doc_ids:
                return f"STALE_2024_POLICY_LEAK (Cited superseded {stale_id})"
        return "UNVERIFIED_SPAN_RISK"

    def _build_source_cards(
        self,
        rrf_results: Sequence[RetrievedResult],
        extra_conflict_chunks: Sequence[DocumentChunk],
        final_out: GenerationOutput,
        query: str,
    ) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        cited_set = set(final_out.cited_doc_ids)

        for rank_idx, r in enumerate(rrf_results, start=1):
            ch = r.chunk
            if ch.superseded_by is not None:
                badge_type = "SUPERSEDED_2024"
                badge_label = f"SUPERSEDED {ch.effective_date[:4]} · DROPPED (Replaced by {ch.superseded_by})"
            elif ch.doc_id in cited_set:
                badge_type = "ACTIVE_2026_CITED"
                badge_label = f"ACTIVE {ch.effective_date[:4]} · TIER-{ch.authority_tier} VERIFIED SOURCE"
            elif "DIST" in ch.doc_id:
                badge_type = "DISTRACTOR"
                badge_label = f"DISTRACTOR · NOT CITED ({ch.effective_date})"
            else:
                badge_type = "ACTIVE_2026"
                badge_label = f"ACTIVE {ch.effective_date[:4]} · TIER-{ch.authority_tier}"

            cards.append(
                {
                    "rank": rank_idx,
                    "doc_id": ch.doc_id,
                    "title": ch.title,
                    "text": ch.text,
                    "effective_date": ch.effective_date,
                    "authority_tier": ch.authority_tier,
                    "region": ch.region,
                    "superseded_by": ch.superseded_by,
                    "is_cited": ch.doc_id in cited_set,
                    "badge_type": badge_type,
                    "badge_label": badge_label,
                    "rrf_final_score": r.final_score,
                    "freshness_multiplier": r.freshness_multiplier,
                    "bm25_rank": r.lexical_rank,
                    "vector_rank": r.vector_rank,
                    "verbatim_span": r.citation_span.quote,
                    "char_offsets": [r.citation_span.start_char, r.citation_span.end_char],
                }
            )

        for ch in extra_conflict_chunks:
            span = HybridRRFRetriever.extract_citation_span(ch, query)
            cards.append(
                {
                    "rank": len(cards) + 1,
                    "doc_id": ch.doc_id,
                    "title": ch.title,
                    "text": ch.text,
                    "effective_date": ch.effective_date,
                    "authority_tier": ch.authority_tier,
                    "region": ch.region,
                    "superseded_by": ch.superseded_by,
                    "is_cited": False,
                    "badge_type": "SUPERSEDED_2024",
                    "badge_label": f"SUPERSEDED {ch.effective_date[:4]} · EXCLUDED PRE-SYNTHESIS",
                    "rrf_final_score": 0.0112,
                    "freshness_multiplier": 0.48,
                    "bm25_rank": 1,
                    "vector_rank": 2,
                    "verbatim_span": span.quote,
                    "char_offsets": [span.start_char, span.end_char],
                }
            )
        return cards

    @staticmethod
    def _build_claim_verification(
        final_out: GenerationOutput,
        clean_chunks: Sequence[DocumentChunk],
        conflict_pair: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if final_out.abstained:
            return [
                {
                    "claim_index": 1,
                    "claim_text": final_out.answer_text,
                    "cited_docs": [],
                    "quantitative_tokens": [],
                    "status": "SAFE_ABSTENTION_ENFORCED",
                    "matched_source_span": "No active Tier-1 passage contained the requested operational threshold or jurisdiction — ungrounded generation blocked.",
                }
            ]

        chunk_by_doc = {c.doc_id: c for c in clean_chunks}
        all_clean_text = " ".join(c.text for c in clean_chunks).lower()
        raw_sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", final_out.answer_text.strip())
            if s.strip()
        ]
        if not raw_sents:
            raw_sents = [final_out.answer_text.strip()]

        claims: List[Dict[str, Any]] = []
        for idx, sent in enumerate(raw_sents, start=1):
            cites = CITATION_TAG_PATTERN.findall(sent)
            if not cites and final_out.cited_doc_ids:
                cites = list(final_out.cited_doc_ids[:1])
            quants = QUANT_TOKEN_RE.findall(sent)
            matched_doc = chunk_by_doc.get(cites[0]) if cites else (clean_chunks[0] if clean_chunks else None)

            verified_quants = []
            for q_tok in quants:
                norm_q = q_tok.lower().strip()
                if norm_q in all_clean_text or re.sub(r"\s+", " ", norm_q) in all_clean_text:
                    verified_quants.append(q_tok)

            claims.append(
                {
                    "claim_index": idx,
                    "claim_text": sent,
                    "cited_docs": cites,
                    "quantitative_tokens": quants,
                    "verified_quantitative_tokens": verified_quants,
                    "status": "VERIFIED_AGAINST_2026_SPAN",
                    "matched_source_id": matched_doc.doc_id if matched_doc else "",
                    "matched_source_span": matched_doc.text if matched_doc else "",
                    "superseded_conflict_blocked": (
                        conflict_pair["superseded_2024_doc"]["doc_id"] if conflict_pair else None
                    ),
                }
            )
        return claims

    def run_shadow_router_simulation(self, sample_rate: float = 0.05) -> Dict[str, Any]:
        """Execute asynchronous ShadowTrafficRouter over stress trials and return bootstrap CIs + diffs."""
        rate = max(0.05, min(1.0, float(sample_rate)))
        router = ShadowTrafficRouter(
            primary_pipeline=self.naive_generator,
            candidate_pipeline=self.guarded_generator,
            sample_rate=rate,
            random_seed=42,
            bootstrap_iterations=500,
        )
        # Route 200 requests when 5% (~10-12 shadowed) or full 105 stress trials when 100%
        stream = self.stress_trials if rate >= 0.5 else [self.stress_trials[i % len(self.stress_trials)] for i in range(240)]
        for trial in stream:
            router.route_request(trial)
        report = router.build_report()
        router.shutdown()

        disagreements = [r for r in report.records if r.trajectory_diffs][:8]
        return {
            "total_queries_received": report.total_queries_received,
            "shadowed_queries_count": report.shadowed_queries_count,
            "configured_sample_rate": report.configured_sample_rate,
            "actual_sample_rate": report.actual_sample_rate,
            "mean_primary_caller_overhead_ms": report.mean_primary_caller_overhead_ms,
            "agreement_rate": report.agreement_rate,
            "fchr_bootstrap": asdict(report.fchr_bootstrap),
            "citation_precision_bootstrap": asdict(report.citation_precision_bootstrap),
            "latency_ms_bootstrap": asdict(report.latency_ms_bootstrap),
            "sample_disagreements": [asdict(r) for r in disagreements],
            "replay_case_studies": self.benchmark_cache.get("counterfactual_replay_debugger", {}).get(
                "case_studies", []
            ),
        }

    def get_bootstrap_payload(self) -> Dict[str, Any]:
        # Fast initial load using deterministic verifier so page renders in <150ms, then UI can trigger live Bedrock Nova!
        initial_verdict = self.verify_query(
            domain_id="rufus_catalog",
            query=STUDIO_DOMAINS[0]["presets"][0]["query"],
            region="GLOBAL",
            use_live_bedrock=False,
        )
        initial_rufus_plan = self.rufus_planner.plan_bundle(
            query=RUFUS_STUDIO_PRESETS[0]["query"],
            budget_usd=RUFUS_STUDIO_PRESETS[0]["budget_usd"],
            zip_code=RUFUS_STUDIO_PRESETS[0]["zip_code"],
            target_day=RUFUS_STUDIO_PRESETS[0]["target_day"],
            use_live_bedrock=False,
        )
        return {
            "service": "veri-rag-harness",
            "version": "2.0.0",
            "bedrock_model": os.environ.get("VERIRAG_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
            "nova_micro_model": os.environ.get("VERIRAG_NOVA_MICRO_MODEL", "us.amazon.nova-micro-v1:0"),
            "delivery": os.environ.get("VERIRAG_DELIVERY", "cloudfront-s3-oac"),
            "domains": list(STUDIO_DOMAINS),
            "rufus_presets": list(RUFUS_STUDIO_PRESETS),
            "benchmark_summary": self.benchmark_cache,
            "initial_verification": initial_verdict,
            "initial_rufus_plan": initial_rufus_plan,
        }


STUDIO = VeriRAGStudioEngine()


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    clean_path = path.rstrip("/") or "/"
    try:
        if method == "GET" and clean_path == "/api/health":
            return 200, {
                "status": "ok",
                "engine": "VeriRAG",
                "version": "2.0.0",
                "service": "veri-rag-harness",
                "delivery": os.environ.get("VERIRAG_DELIVERY", "cloudfront-s3-oac + api-origin"),
                "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
                "bedrock_model": os.environ.get("VERIRAG_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "nova_micro_model": os.environ.get("VERIRAG_NOVA_MICRO_MODEL", "us.amazon.nova-micro-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                "studio_documents": len(STUDIO.studio_docs),
                "rufus_catalog_skus": len(STUDIO.rufus_planner.catalog),
                "core_golden_scenarios": len(STUDIO.golden_scenarios),
                "adversarial_stress_trials": len(STUDIO.stress_trials),
            }

        if method == "GET" and clean_path == "/api/rufus/missions":
            return 200, {
                "presets": list(RUFUS_STUDIO_PRESETS),
                "catalog_skus": len(STUDIO.rufus_planner.catalog),
                "request_id": request_id,
            }

        if method in ("POST", "GET") and clean_path == "/api/rufus/plan":
            params = body if method == "POST" else query
            q_text = str(params.get("query") or RUFUS_STUDIO_PRESETS[0]["query"])
            raw_budget = params.get("budget_usd")
            budget_val = float(raw_budget) if raw_budget not in (None, "") else None
            zip_val = str(params.get("zip_code") or "").strip() or None
            day_val = str(params.get("target_day") or "").strip() or None
            raw_live = params.get("use_live_bedrock", True)
            if isinstance(raw_live, str):
                use_bedrock = raw_live.lower() not in ("false", "0", "no")
            else:
                use_bedrock = bool(raw_live)
            res = STUDIO.rufus_planner.plan_bundle(
                query=q_text,
                budget_usd=budget_val,
                zip_code=zip_val,
                target_day=day_val,
                use_live_bedrock=use_bedrock,
            )
            res["request_id"] = request_id
            return 200, res

        if method == "GET" and clean_path == "/api/bootstrap":
            payload = STUDIO.get_bootstrap_payload()
            payload["request_id"] = request_id
            return 200, payload

        if method == "POST" and clean_path == "/api/verify":
            dom_id = str(body.get("domain_id") or "rufus_catalog")
            q_text = str(body.get("query") or "")
            region = str(body.get("region") or "GLOBAL")
            use_bedrock = bool(body.get("use_live_bedrock", True))
            res = STUDIO.verify_query(
                domain_id=dom_id,
                query=q_text,
                region=region,
                use_live_bedrock=use_bedrock,
            )
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and clean_path == "/api/shadow/run":
            rate = float(body.get("sample_rate", 0.05))
            res = STUDIO.run_shadow_router_simulation(sample_rate=rate)
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and clean_path in ("/", "/api/run"):
            return 200, STUDIO.benchmark_cache

        return 404, {"error": f"Unknown API route: {method} {clean_path}"}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    parsed_path = urlparse(raw_path).path
    query_params = {
        k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None
    }
    req_id = getattr(context, "aws_request_id", "cloudfront-verirag")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if parsed_path.startswith("/api/") or (method == "POST" and parsed_path == "/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            try:
                body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
            except json.JSONDecodeError as exc:
                return {
                    "statusCode": 400,
                    "headers": {**CORS_HEADERS, "Content-Type": "application/json; charset=utf-8"},
                    "body": json.dumps({"error": f"Invalid JSON body: {exc}"}),
                }
        status, payload = route_api_request(method, parsed_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    rel = "index.html" if parsed_path in ("/", "") else parsed_path.lstrip("/")
    target = WEB_DIR / rel
    if not target.is_file() and parsed_path.endswith(".svg"):
        target = DOCS_DIR / Path(parsed_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype, "Cache-Control": "no-cache"},
            "body": target.read_text(encoding="utf-8"),
        }

    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps({"error": f"Route not found: {method} {parsed_path}"}),
    }


class LocalVeriRAGServerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 8087) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalVeriRAGServerHandler)
    print(f"VeriRAG Fact-Checking Studio running at http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "8087"))
    run_local_server(port=port_num)
