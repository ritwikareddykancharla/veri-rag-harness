"""Unit & Integration tests for Rufus Multi-Step Bundle & Constraint Planner (`tests/test_rufus_planner.py`)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from veri_rag.rufus_planner import (
    RufusAgenticPlanner,
    build_rufus_catalog,
    build_rufus_mission_benchmark_scenarios,
    evaluate_rufus_planner_benchmark,
)


class TestRufusAgenticPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = build_rufus_catalog()
        cls.planner = RufusAgenticPlanner(catalog=cls.catalog)

    def test_01_catalog_contains_three_families_and_spec_conflicts(self) -> None:
        self.assertGreaterEqual(len(self.catalog), 24)
        families = {s.mission_family for s in self.catalog}
        self.assertEqual(
            families,
            {"mirrorless_travel_kit", "usb_c_desk_setup", "pourover_coffee_bar"},
        )
        conflict_skus = [s for s in self.catalog if s.seller_claim_conflict]
        self.assertGreaterEqual(len(conflict_skus), 6)

    def test_02_socket_mount_compatibility_blocks_rf_mount_and_usb2_cable_traps(self) -> None:
        by_asin = {s.asin: s for s in self.catalog}
        # Trap 1: Canon RF lens on Sony E-mount host + proprietary tripod clip + 185Wh battery
        bad_cam = self.planner.check_socket_mount_compatibility(
            [by_asin["B0CNRF3518"], by_asin["B0TRP99CLP"], by_asin["B0PWR180WH"], by_asin["B0SD128TF1"]],
            {
                "host_lens_mount": "SONY_E",
                "required_tripod_plate": "ARCA_SWISS_38MM",
                "faa_carry_on_max_wh": 100.0,
                "min_video_write_mb_s": 60,
            },
        )
        self.assertFalse(bad_cam["compatible"])
        self.assertEqual(len(bad_cam["violations"]), 4)

        # Trap 2: USB 2.0 charge-only cable + 48V XLR mic on USB-C 4K60 desk setup
        bad_desk = self.planner.check_socket_mount_compatibility(
            [by_asin["B0DCK10PD8"], by_asin["B0MON274K6"], by_asin["B0ARM100VS"], by_asin["B0CBL02CHG"], by_asin["B0MIC48XLR"]],
            {"host_port": "USB4_TB4", "require_4k60_video": True, "min_host_pd_watts": 85},
        )
        self.assertFalse(bad_desk["compatible"])
        self.assertGreaterEqual(len(bad_desk["violations"]), 2)

    def test_03_prime_zip_sla_blocks_over_100wh_air_hazmat_and_slow_ltl(self) -> None:
        by_asin = {s.asin: s for s in self.catalog}
        hazmat_sla = self.planner.verify_prime_zip_sla(by_asin["B0PWR180WH"], "98109", "Friday")
        self.assertFalse(hazmat_sla["meets_sla"])
        self.assertIn("DOC-SSD-2026-03", hazmat_sla["reason"])

        legal_sla = self.planner.verify_prime_zip_sla(by_asin["B09X7RUFS1"], "98109", "Friday")
        self.assertTrue(legal_sla["meets_sla"])

        ltl_sla = self.planner.verify_prime_zip_sla(by_asin["B0MON32HVY"], "98109", "Friday")
        self.assertFalse(ltl_sla["meets_sla"])

    def test_04_end_to_end_desk_and_mirrorless_bundles_under_450_seattle(self) -> None:
        with patch.dict(os.environ, {"VERIRAG_OFFLINE_LLM": "1"}):
            desk = self.planner.plan_bundle(
                query=(
                    "Build a complete home-office USB-C desk setup under $450 delivered to Seattle 98109 "
                    "by Friday where every cable and mount is compatible."
                ),
                budget_usd=450.0,
                zip_code="98109",
                target_day="Friday",
                use_live_bedrock=False,
            )
            self.assertTrue(desk["feasible"])
            self.assertLessEqual(desk["total_price_usd"], 450.0)
            self.assertTrue(desk["compatibility_report"]["compatible"])
            desk_asins = [i["asin"] for i in desk["bundle_items"]]
            self.assertEqual(len(desk_asins), 5)
            self.assertIn("B0CBL40DP4", desk_asins)
            self.assertNotIn("B0CBL02CHG", desk_asins)
            self.assertNotIn("B0MIC48XLR", desk_asins)

            cam = self.planner.plan_bundle(
                query=(
                    "Build a complete mirrorless travel kit for Sony E-mount under $450 delivered to "
                    "Seattle 98109 by Friday with FAA carry-on battery and Arca-Swiss tripod."
                ),
                budget_usd=450.0,
                zip_code="98109",
                target_day="Friday",
                use_live_bedrock=False,
            )
            self.assertTrue(cam["feasible"])
            self.assertLessEqual(cam["total_price_usd"], 450.0)
            cam_asins = [i["asin"] for i in cam["bundle_items"]]
            self.assertIn("B0SNYE3518", cam_asins)
            self.assertIn("B09X7RUFS1", cam_asins)
            self.assertNotIn("B0CNRF3518", cam_asins)
            self.assertNotIn("B0PWR180WH", cam_asins)

    def test_05_tiered_routing_and_tier1_spec_hallucination_gate(self) -> None:
        invoked_models: list[str] = []

        def mock_bedrock(model_id: str, sys_p: str, usr_p: str, max_tok: int) -> dict[str, object]:
            invoked_models.append(model_id)
            if "nova-micro" in model_id:
                return {
                    "text": '{"mission_family":"mirrorless_travel_kit","budget_usd":450,"zip_code":"98109","target_day":"Friday","is_underspecified":false}',
                    "model_id": model_id,
                    "latency_ms": 92.0,
                    "input_tokens": 110,
                    "output_tokens": 42,
                    "live_bedrock": True,
                }
            # Simulate LLM attempting to leak stale 2024 seller claim "120 Wh" and "$59" + fake ASIN
            return {
                "text": "Buy [B09X7RUFS1] with 120 Wh battery at $59 and fake [B0FAKE9999]!",
                "model_id": model_id,
                "latency_ms": 210.0,
                "input_tokens": 240,
                "output_tokens": 55,
                "live_bedrock": True,
            }

        custom_planner = RufusAgenticPlanner(
            catalog=self.catalog,
            nova_micro_model="us.amazon.nova-micro-v1:0",
            nova_lite_model="us.amazon.nova-lite-v1:0",
            custom_llm_caller=mock_bedrock,
        )
        res = custom_planner.plan_bundle(
            query="Build a complete Sony E-mount mirrorless travel kit under $450 to Seattle 98109 by Friday.",
            use_live_bedrock=True,
        )
        self.assertEqual(invoked_models, ["us.amazon.nova-micro-v1:0", "us.amazon.nova-lite-v1:0"])
        self.assertTrue(res["synthesis"]["gate_repaired"])
        self.assertNotIn("B0FAKE9999", res["synthesis"]["text"])
        self.assertIn("99.9 Wh", res["synthesis"]["text"])

    def test_06_infeasible_budget_and_underspecified_clarification_guardrails(self) -> None:
        infeasible = self.planner.plan_bundle(
            query="Build a complete 5-item USB-C 4K60 monitor, arm, dock, cable, and mic desk setup under $95 to 98109.",
            budget_usd=95.0,
            zip_code="98109",
            target_day="Friday",
            use_live_bedrock=False,
        )
        self.assertFalse(infeasible["feasible"])
        self.assertTrue(infeasible["abstained"])
        self.assertEqual(infeasible["status"], "INFEASIBLE_BUDGET")
        self.assertGreater(infeasible["min_compatible_bundle_usd"], 350.0)

        clarify = self.planner.plan_bundle(
            query="Buy something nice for a gift.",
            use_live_bedrock=False,
        )
        self.assertTrue(clarify["abstained"])
        self.assertEqual(clarify["status"], "CLARIFICATION_REQUIRED")
        self.assertIn("CLARIFY:", clarify["synthesis"]["text"])

    def test_07_twenty_four_mission_composite_benchmark_significance(self) -> None:
        scenarios = build_rufus_mission_benchmark_scenarios()
        self.assertEqual(len(scenarios), 24)
        report = evaluate_rufus_planner_benchmark(self.planner)
        cand = report["candidate_rufus_planner_v2"]
        base = report["baseline_naive_shop_rag"]
        self.assertEqual(cand["composite_mission_success_rate"], 1.0)
        self.assertEqual(cand["physical_compatibility_rate"], 1.0)
        self.assertEqual(cand["spec_price_hallucination_rate"], 0.0)
        self.assertEqual(cand["safe_refusal_clarification_recall"], 1.0)
        self.assertGreater(cand["composite_mission_success_rate"], base["composite_mission_success_rate"] + 0.40)
        self.assertGreater(cand["mean_parallel_speedup_x"], 1.5)
        self.assertTrue(report["composite_success_bootstrap"]["is_statistically_significant"])


if __name__ == "__main__":
    unittest.main()
