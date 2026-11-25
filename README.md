# VeriRAG v2: Rufus Multi-Step Bundle Planner & Policy Citation Verifier

**Live Production App (Amazon CloudFront + Private S3 OAC + `/api/*` + Bedrock Nova Micro/Lite):** https://d14kv31n2lrkoa.cloudfront.net  
**Health Check:** `https://d14kv31n2lrkoa.cloudfront.net/api/health`  
**Live Rufus Planner API:** `curl -s -X POST https://d14kv31n2lrkoa.cloudfront.net/api/rufus/plan -H 'Content-Type: application/json' -d '{"query":"Build a complete home-office USB-C desk setup under $450 delivered to Seattle 98109 by Friday where every cable and mount is compatible.","budget_usd":450,"zip_code":"98109","target_day":"Friday"}'`  
**Live Policy Verify API:** `curl -s -X POST https://d14kv31n2lrkoa.cloudfront.net/api/verify -H 'Content-Type: application/json' -d '{"domain_id":"rufus_catalog","query":"Is the HydroVolt 27K power bank (B09X7RUFS1) FAA carry-on legal under 100 Wh?"}'`

**VeriRAG v2** combines version-aware policy search and span-level citation verification with a **Multi-Step Agentic Rufus Shopping Bundle & Compatibility Planner**. It decomposes conversational multi-item shopping missions via **Amazon Bedrock (`us.amazon.nova-micro-v1:0`)**, executes `search_catalog`, `check_socket_mount_compatibility`, `verify_prime_zip_sla`, and `solve_budget_bundle` concurrently (`4.5x` parallel speedup, `4.52 ms` p50), synthesizes grounded carts via **Amazon Bedrock (`us.amazon.nova-lite-v1:0`)** (`92.6%` schema token pruning, `-98.0%` inference cost), enforces Tier-1 UL/OEM/USB-IF spec and price gates (`50.0% -> 100.0%` composite mission success across `24` shopping missions), ranks active `2026` policies above superseded `2024` documents (`75.0% -> 100.0%` Top-1 hit rate), abstains cleanly when evidence is missing (`73.3% -> 0.0%` false-confidence hallucinations), and mirrors `5%` of traffic asynchronously (`< 0.001 ms` caller overhead).

![VeriRAG CloudFront Architecture](docs/architecture.svg)

---

## What You Can Do Live in the Web Studio

1. **Rufus Multi-Step Bundle & Compatibility Planner (`/api/rufus/plan`)**:
   - Plans complete multi-item shopping kits (Sony E-mount vs. Canon RF mirrorless travel kits, home-office USB-C 4K60 VESA desk setups, and 120V narrow-counter pour-over coffee bars) within hard dollar caps and Prime ZIP arrival cutoffs (`98109` Seattle, `10001` NYC, `78701` Austin, `99501` Anchorage air-linehaul).
   - Verifies pairwise physical mounts, DisplayPort Alt-Mode 4K60 cables, FAA `<= 100 Wh` lithium rules, and active Tier-1 catalog prices while rejecting incompatible or exaggerated 3P seller listings and abstaining cleanly on impossible budgets.
2. **3-Domain Grounded Commerce & Policy Fact-Checking Studio (`/api/verify`)**:
   - **Rufus Catalog Spec Verifier**: Resolves conflicting 3P seller listing bullets vs. verified UL/ISO manufacturer spec sheets (`99.9 Wh` FAA-legal vs. `120 Wh` seller bullet).
   - **Seller Assistant (SPX) 2024 vs. 2026 Resolver**: Displays side-by-side superseded `2024` PDFs vs. active `2026` FBA reimbursement (`$5,000` vs. `$2,000`), SAFE-T (`45` vs. `60` days), and SFP (`97.5%` vs. `93.5%`) policies.
   - **Customer Service & Clinical Safe Abstention**: Blocks unsupported returnless refund promises (`>$75`) and unverified clinical dosing thresholds.
3. **5% Async Shadow Router & 5-Stage Counterfactual Replay Debugger**:
   - Inspect paired bootstrap `95%` confidence intervals (`1,000` iterations) and swap individual pipeline stages using `SHA-256` state checkpoints in `0 ms`.

---

## Measured Benchmark Results (`benchmarks/latest_results.json`)

### 1. Rufus Multi-Step Bundle & Constraint Planner (`24` Shopping Missions)

| Metric | Baseline (`NaiveSequentialShopRAG`) | `VeriRAG-RufusPlanner-v2` | Delta / Significance |
| :--- | :---: | :---: | :---: |
| **Composite Shopping Mission Success Rate** | 50.0% | **100.0%** | **+50.0% ± 18.8%** (`95% CI [+33.3%, +70.8%]`, $p = 0.001$) |
| **Physical Socket & Mount Compatibility Rate** | 72.2% | **100.0%** | **+27.8%** (`0` mismatched mounts/cables) |
| **Hard Budget & Prime ZIP SLA Compliance** | 83.3% | **100.0%** | **+16.7%** (exact knapsack + air hazmat check) |
| **3P Seller Spec & Stale Price Hallucination Rate** | 50.0% | **0.0%** | **-50.0%** (Tier-1 UL/OEM/USB-IF spec gate) |
| **Infeasible / Ambiguous Mission Safe Refusal Recall** | 0.0% | **100.0%** | **+100.0%** (`6/6` guardrail missions caught) |
| **Parallel Tool Orchestration Latency (`p50` / `p95`)** | 21.74 ms | **4.52 ms / 4.69 ms** | **4.5x parallel fan-out speedup** |
| **Inference Cost per 1,000 Missions** | \$3.3830 | **\$0.0673** | **-98.0% cost · 92.6% schema token pruning** |

### 2. Hybrid Retrieval, Adversarial Abstention & Shadow Routing (`105` Perturbed Trials)

| Metric | Baseline (`BM25` / Naive Top-K RAG) | VeriRAG (Date-Ranked RRF + Span Verifier) | Delta / Significance |
| :--- | :---: | :---: | :---: |
| **Top-1 Retrieval Hit Rate (Conflicting Policies)** | 75.0% (`MRR@3 = 0.88`) | **100.0%** (`MRR@3 = 1.00`, `nDCG@3 = 1.00`) | **+25.0%** |
| **2026 vs. 2024 Policy Conflict Accuracy** | 41.7% to 50.0% | **100.0%** | **+50.0% to +58.3%** |
| **Unsupported Answer Rate (105 Perturbed Trials)** | 73.3% | **0.0%** | **-73.3%** (`95% CI [-81.0%, -64.8%]`, $p = 0.001$) |
| **Citation Precision** | 40.3% | **100.0%** | **+60.0%** (`95% CI [+51.0%, +69.0%]`, $p = 0.001$) |
| **Abstention Recall / Precision on Missing Evidence** | 0.0% / 0.0% | **100.0% recall** / **95.7% precision** | Cleanly abstains when evidence is absent |
| **Async 5% Shadow Traffic Caller Overhead** | N/A | **`< 0.001 ms`** | Background `ThreadPoolExecutor` |

---

## Run Locally & Deploy to Amazon CloudFront

```bash
# 1. Run all 34 unit & integration tests + deterministic benchmarks
PYTHONPATH=. python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 benchmarks/run_benchmarks.py

# 2. Start the interactive studio locally
PYTHONPATH=. python3 -m veri_rag.web_api
# -> Open http://127.0.0.1:8087

# 3. Deploy to Amazon CloudFront + Private S3 (OAC) + /api/* + Amazon Bedrock (us-west-2)
chmod +x infra/deploy_cloudfront.sh
./infra/deploy_cloudfront.sh
```
