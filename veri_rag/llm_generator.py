"""
Multi-Provider Live LLM Generator (`LiveLLMVeriRAGGenerator`) for VeriRAG.

Supports:
  - Anthropic (`ANTHROPIC_API_KEY`, default model `claude-opus-5-5`, header `anthropic-version: 2026-10-21`)
  - OpenAI (`OPENAI_API_KEY`, default model `gpt-4o-mini`)
  - Google Gemini (`GEMINI_API_KEY`, default model `gemini-2.5-flash`)
  - Amazon Bedrock (`AWS_BEDROCK_MODEL_ID` via optional `boto3`)
  - Automatic deterministic fallback to `VeriRAGGenerator` when no API key is present.

Enforces pre-synthesis supersession conflict resolution, strict `[DOC-ID]` inline
citation grounding verification, and post-generation abstention enforcement using
pure Python standard library (`urllib.request`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional
import urllib.error
import urllib.request

from veri_rag.adversarial_harness import (
    CITATION_TAG_PATTERN,
    GenerationOutput,
    VeriRAGGenerator,
)
from veri_rag.hybrid_retriever import DocumentChunk


SYSTEM_INSTRUCTION = (
    "You are VeriRAG, a high-assurance enterprise policy and clinical protocol "
    "compliance synthesizer. Follow these rules strictly:\n"
    "1. Answer ONLY using facts explicitly stated in the provided VERIFIED CONTEXT CHUNKS.\n"
    "2. Cite every factual claim inline using exact bracketed document IDs, e.g. [DOC-SSD-2026-01].\n"
    "3. Never cite a document ID that is not in the provided context chunks.\n"
    "4. If two documents conflict, strictly enforce the newer effective_date and Tier-1 authority, "
    "and ignore superseded documents.\n"
    "5. If the context chunks do not contain the specific quantitative threshold, jurisdiction, "
    "or operational condition asked in the query, respond with exact prefix: "
    "'ABSTAIN: <concise reason>'."
)


def load_dotenv_if_present() -> None:
    """
    Loads environment variables from `.env.local` in current or parent workspace
    directories into `os.environ` without overwriting existing process variables.
    Never logs or returns secret values.
    """
    candidates = [
        Path.cwd() / ".env.local",
        Path.cwd().parent / ".env.local",
        Path.cwd().parent.parent / ".env.local",
        *[p / ".env.local" for p in Path(__file__).resolve().parents],
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    k = key.strip()
                    v = val.strip().strip("'").strip('"')
                    if k and k not in os.environ:
                        os.environ[k] = v
            except OSError:
                pass


class LiveLLMVeriRAGGenerator:
    """
    Multi-provider LLM RAG synthesizer with pre-prompt conflict filtering,
    post-generation citation verification, and deterministic fallback.
    """

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout_sec: float = 30.0,
        allow_fallback: bool = True,
        custom_transport: Optional[Callable[[str, str], str]] = None,
        auto_load_dotenv: bool = False,
    ) -> None:
        if auto_load_dotenv:
            load_dotenv_if_present()
        self.provider = self._resolve_provider(provider)
        self.model = model or self._default_model_for_provider(self.provider)
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.allow_fallback = allow_fallback
        self.custom_transport = custom_transport
        self._fallback_generator = VeriRAGGenerator()
        self.name = f"LiveLLMVeriRAG ({self.provider}:{self.model})"
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _resolve_provider(requested: str) -> str:
        req = requested.lower().strip()
        if req != "auto":
            return req
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("AWS_BEDROCK_MODEL_ID"):
            return "bedrock"
        return "deterministic_fallback"

    @staticmethod
    def _default_model_for_provider(provider: str) -> str:
        defaults = {
            "anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"),
            "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "gemini": "gemini-2.5-flash",
            "bedrock": os.environ.get(
                "AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"
            ),
            "deterministic_fallback": "veri-rag-deterministic-v1",
        }
        return defaults.get(provider, "veri-rag-deterministic-v1")

    @staticmethod
    def build_grounded_prompt(
        query: str,
        clean_chunks: list[DocumentChunk],
        dropped_superseded_ids: list[str],
    ) -> str:
        """Constructs a structured, provenance-tagged prompt for the LLM."""
        context_blocks: list[str] = []
        for idx, ch in enumerate(clean_chunks, start=1):
            block = (
                f"--- CHUNK {idx} ---\n"
                f"DOC_ID: [{ch.doc_id}]\n"
                f"TITLE: {ch.title}\n"
                f"DOMAIN: {ch.domain} | REGION: {ch.region} | "
                f"EFFECTIVE_DATE: {ch.effective_date} | AUTHORITY_TIER: {ch.authority_tier}\n"
                f"CONTENT: {ch.text}"
            )
            context_blocks.append(block)

        superseded_note = (
            f"\nNOTE: Superseded/untrusted documents automatically excluded prior to synthesis: "
            f"{', '.join(dropped_superseded_ids)}\n"
            if dropped_superseded_ids
            else ""
        )

        context_str = "\n\n".join(context_blocks) if context_blocks else "(NONE)"
        return (
            f"VERIFIED CONTEXT CHUNKS:\n{context_str}\n"
            f"{superseded_note}\n"
            f"USER QUERY: {query}\n\n"
            "Provide a concise, grounded response with inline [DOC-ID] citations, "
            "or output 'ABSTAIN: <reason>' if the evidence is insufficient or out-of-scope."
        )

    @staticmethod
    def verify_and_parse_llm_response(
        raw_response: str,
        clean_chunks: list[DocumentChunk],
        conflict_detected: bool,
        dropped_docs: list[str],
        latency_ms: float,
        cost_usd: float = 0.00065,
    ) -> GenerationOutput:
        """
        Verifies that the LLM response either abstains cleanly or cites only valid
        document IDs from `clean_chunks`. Strips any hallucinated/spoofed citations
        and forces abstention if a non-abstained answer has zero valid citations.
        """
        text = raw_response.strip()
        allowed_doc_ids = {c.doc_id for c in clean_chunks}
        chunk_by_doc = {c.doc_id: c for c in clean_chunks}

        if text.upper().startswith("ABSTAIN"):
            reason = text.split(":", 1)[1].strip() if ":" in text else "llm_abstained"
            return GenerationOutput(
                answer_text=text if text.startswith("ABSTAIN:") else f"ABSTAIN: {reason}",
                abstained=True,
                abstention_reason=reason,
                cited_doc_ids=[],
                selected_chunk_ids=[],
                conflict_detected=conflict_detected,
                superseded_docs_dropped=dropped_docs,
                grounding_score=1.0,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost_usd, 6),
            )

        extracted_citations = CITATION_TAG_PATTERN.findall(text)
        valid_citations: list[str] = []
        invalid_citations: list[str] = []

        for doc_id in extracted_citations:
            if doc_id in allowed_doc_ids:
                if doc_id not in valid_citations:
                    valid_citations.append(doc_id)
            else:
                invalid_citations.append(doc_id)

        if invalid_citations or not valid_citations:
            return GenerationOutput(
                answer_text=(
                    "ABSTAIN: Post-generation citation guard rejected response due to "
                    f"unverified or missing citations (valid={valid_citations}, "
                    f"invalid={invalid_citations})."
                ),
                abstained=True,
                abstention_reason="post_generation_citation_guard_triggered",
                cited_doc_ids=[],
                selected_chunk_ids=[],
                conflict_detected=conflict_detected,
                superseded_docs_dropped=dropped_docs,
                grounding_score=0.0,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost_usd, 6),
            )

        selected_chunks = [
            chunk_by_doc[d].chunk_id for d in valid_citations if d in chunk_by_doc
        ]
        return GenerationOutput(
            answer_text=text,
            abstained=False,
            abstention_reason=None,
            cited_doc_ids=valid_citations,
            selected_chunk_ids=selected_chunks,
            conflict_detected=conflict_detected,
            superseded_docs_dropped=dropped_docs,
            grounding_score=1.0,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6),
        )

    def _call_anthropic(self, prompt: str) -> tuple[str, dict[str, int]]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        anth_version = os.environ.get("ANTHROPIC_VERSION", "2026-10-21")
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": 640,
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": anth_version,
                "user-agent": "curl/8.7.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        usage = body.get("usage", {})
        tokens = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }
        # Extract text block
        text_parts = [
            blk.get("text", "")
            for blk in body.get("content", [])
            if blk.get("type") == "text"
        ]
        return ("\n".join(text_parts) if text_parts else body["content"][0]["text"]), tokens

    def _call_openai(self, prompt: str) -> tuple[str, dict[str, int]]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 320,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        usage = body.get("usage", {})
        tokens = {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }
        return body["choices"][0]["message"]["content"], tokens

    def _call_gemini(self, prompt: str) -> tuple[str, dict[str, int]]:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": 320},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["candidates"][0]["content"]["parts"][0]["text"], {
            "input_tokens": 220,
            "output_tokens": 60,
        }

    def _call_bedrock(self, prompt: str) -> tuple[str, dict[str, int]]:
        import boto3  # type: ignore

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        response = client.converse(
            modelId=self.model,
            system=[{"text": SYSTEM_INSTRUCTION}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 320, "temperature": self.temperature},
        )
        return response["output"]["message"]["content"][0]["text"], {
            "input_tokens": 220,
            "output_tokens": 60,
        }

    def generate(self, query: str, chunks: list[DocumentChunk]) -> GenerationOutput:
        t0 = time.perf_counter()

        ood_reason = VeriRAGGenerator._detect_ood_boundary(query, chunks)
        if ood_reason is not None:
            fb_out = self._fallback_generator.generate(query, chunks)
            self.last_trace = {
                "provider": self.provider,
                "model": self.model,
                "gate_triggered": ood_reason,
                "raw_llm_response": fb_out.answer_text,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": fb_out.latency_ms,
            }
            return fb_out

        clean_chunks, conflict_detected, dropped_docs = (
            VeriRAGGenerator._resolve_conflicts(chunks)
        )
        if not clean_chunks:
            fb_out = self._fallback_generator.generate(query, chunks)
            self.last_trace = {
                "provider": self.provider,
                "model": self.model,
                "gate_triggered": "all_chunks_superseded",
                "raw_llm_response": fb_out.answer_text,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": fb_out.latency_ms,
            }
            return fb_out

        prompt = self.build_grounded_prompt(query, clean_chunks, dropped_docs)

        if self.custom_transport is not None:
            raw_text = self.custom_transport(SYSTEM_INSTRUCTION, prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out = self.verify_and_parse_llm_response(
                raw_text, clean_chunks, conflict_detected, dropped_docs, elapsed_ms
            )
            self.last_trace = {
                "provider": self.provider,
                "model": self.model,
                "prompt_sent": prompt,
                "raw_llm_response": raw_text,
                "input_tokens": 200,
                "output_tokens": 45,
                "latency_ms": out.latency_ms,
            }
            return out

        if self.provider == "deterministic_fallback":
            fb_out = self._fallback_generator.generate(query, chunks)
            self.last_trace = {
                "provider": "deterministic_fallback",
                "model": self.model,
                "prompt_sent": prompt,
                "raw_llm_response": fb_out.answer_text,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": fb_out.latency_ms,
            }
            return fb_out

        try:
            if self.provider == "anthropic":
                raw_text, tok = self._call_anthropic(prompt)
                cost = (tok["input_tokens"] * 0.000005) + (tok["output_tokens"] * 0.000025)
            elif self.provider == "openai":
                raw_text, tok = self._call_openai(prompt)
                cost = (tok["input_tokens"] * 0.00000015) + (tok["output_tokens"] * 0.0000006)
            elif self.provider == "gemini":
                raw_text, tok = self._call_gemini(prompt)
                cost = 0.00035
            elif self.provider == "bedrock":
                raw_text, tok = self._call_bedrock(prompt)
                cost = 0.00045
            else:
                return self._fallback_generator.generate(query, chunks)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out = self.verify_and_parse_llm_response(
                raw_text,
                clean_chunks,
                conflict_detected,
                dropped_docs,
                elapsed_ms,
                cost_usd=cost,
            )
            self.last_trace = {
                "provider": self.provider,
                "model": self.model,
                "prompt_sent": prompt,
                "raw_llm_response": raw_text,
                "input_tokens": tok["input_tokens"],
                "output_tokens": tok["output_tokens"],
                "latency_ms": out.latency_ms,
                "cost_usd": out.cost_usd,
            }
            return out
        except Exception as exc:
            if self.allow_fallback:
                fb_out = self._fallback_generator.generate(query, chunks)
                self.last_trace = {
                    "provider": f"{self.provider}_fallback",
                    "model": self.model,
                    "error": type(exc).__name__,
                    "prompt_sent": prompt,
                    "raw_llm_response": fb_out.answer_text,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": fb_out.latency_ms,
                }
                return fb_out
            raise
