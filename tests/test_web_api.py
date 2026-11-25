"""Integration tests for VeriRAG 3-Domain Grounding Studio & Shadow Router API (`tests/test_web_api.py`)."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from veri_rag.web_api import lambda_handler


class TestVeriRAGWebApi(unittest.TestCase):
    def test_health_and_bootstrap(self) -> None:
        h = lambda_handler({"httpMethod": "GET", "rawPath": "/api/health"})
        self.assertEqual(h["statusCode"], 200)
        hb = json.loads(h["body"])
        self.assertEqual(hb["status"], "ok")
        self.assertEqual(hb["bedrock_model"], "us.amazon.nova-lite-v1:0")

        b = lambda_handler({"httpMethod": "GET", "rawPath": "/api/bootstrap"})
        self.assertEqual(b["statusCode"], 200)
        bb = json.loads(b["body"])
        self.assertEqual(len(bb["domains"]), 3)
        self.assertFalse(bb["initial_verification"]["answer"]["abstained"])

    def test_three_domains_conflict_and_abstention(self) -> None:
        with patch.dict(os.environ, {"VERIRAG_OFFLINE_LLM": "1"}):
            # 1. Rufus Catalog Spec conflict resolution (99.9 Wh 2026 UL spec vs 120 Wh 2024 seller bullet)
            r_rufus = lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/verify",
                    "body": json.dumps(
                        {
                            "domain_id": "rufus_catalog",
                            "query": "Is the HydroVolt 27K power bank (B09X7RUFS1) FAA carry-on legal under 100 Wh?",
                            "use_live_bedrock": False,
                        }
                    ),
                }
            )
            self.assertEqual(r_rufus["statusCode"], 200)
            b_rufus = json.loads(r_rufus["body"])
            self.assertFalse(b_rufus["answer"]["abstained"])
            self.assertIn("DOC-CAT-2026-01", b_rufus["answer"]["cited_doc_ids"])
            self.assertIn("DOC-CAT-2024-01", b_rufus["answer"]["superseded_docs_dropped"])
            self.assertIsNotNone(b_rufus["conflict_side_by_side"])

            # 2. Seller SPX 2024 ($2,000) vs 2026 ($5,000) FBA reimbursement cap
            r_spx = lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/verify",
                    "body": json.dumps(
                        {
                            "domain_id": "seller_spx",
                            "query": "What is the automated Seller Support reimbursement valuation cap per single unit for FBA lost-in-warehouse inventory?",
                            "region": "US-EAST",
                            "use_live_bedrock": False,
                        }
                    ),
                }
            )
            b_spx = json.loads(r_spx["body"])
            self.assertFalse(b_spx["answer"]["abstained"])
            self.assertIn("DOC-SPS-2026-01", b_spx["answer"]["cited_doc_ids"])
            self.assertIn("$5,000", b_spx["answer"]["text"])

            # 3. Safe Abstention on unverified clinical dosage
            r_abs = lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/verify",
                    "body": json.dumps(
                        {
                            "domain_id": "cs_clinical_safety",
                            "query": "What is the approved intrathecal bolus mg/kg dosage for neonate infants under 4 kg with acute renal dialysis?",
                            "use_live_bedrock": False,
                        }
                    ),
                }
            )
            b_abs = json.loads(r_abs["body"])
            self.assertTrue(b_abs["answer"]["abstained"])
            self.assertEqual(b_abs["claims_breakdown"][0]["status"], "SAFE_ABSTENTION_ENFORCED")

    def test_shadow_router_endpoint(self) -> None:
        r_sh = lambda_handler(
            {
                "httpMethod": "POST",
                "rawPath": "/api/shadow/run",
                "body": json.dumps({"sample_rate": 0.05}),
            }
        )
        self.assertEqual(r_sh["statusCode"], 200)
        b_sh = json.loads(r_sh["body"])
        self.assertGreater(b_sh["shadowed_queries_count"], 0)
        self.assertLess(b_sh["fchr_bootstrap"]["mean_delta"], 0.0)

    def test_rufus_planner_endpoint(self) -> None:
        with patch.dict(os.environ, {"VERIRAG_OFFLINE_LLM": "1"}):
            r_missions = lambda_handler({"httpMethod": "GET", "rawPath": "/api/rufus/missions"})
            self.assertEqual(r_missions["statusCode"], 200)
            bm = json.loads(r_missions["body"])
            self.assertEqual(len(bm["presets"]), 4)

            r_plan = lambda_handler(
                {
                    "httpMethod": "POST",
                    "rawPath": "/api/rufus/plan",
                    "body": json.dumps(
                        {
                            "query": "Build a complete home-office USB-C desk setup under $450 delivered to Seattle 98109 by Friday.",
                            "budget_usd": 450,
                            "zip_code": "98109",
                            "target_day": "Friday",
                            "use_live_bedrock": False,
                        }
                    ),
                }
            )
            self.assertEqual(r_plan["statusCode"], 200)
            bp = json.loads(r_plan["body"])
            self.assertTrue(bp["feasible"])
            self.assertLessEqual(bp["total_price_usd"], 450.0)
            self.assertTrue(bp["compatibility_report"]["compatible"])
            self.assertEqual(len(bp["bundle_items"]), 5)


if __name__ == "__main__":
    unittest.main()
