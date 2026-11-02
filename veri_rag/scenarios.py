"""
Enterprise Multi-Document Corpus (54 documents) and 30 Golden Evaluation Scenarios
spanning three target domains:
  1. Same-Day Delivery SLA & Routing Policies (Amazon WW Speed & AI Tech / Same Day Delivery)
  2. Selling Partner Support & Seller Assistant Reimbursement Policies (Amazon SPS)
  3. Clinical & Regulatory Protocol Compliance (AWS Applied AI - Life Sciences)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from veri_rag.hybrid_retriever import Document


@dataclass(frozen=True)
class EvaluationScenario:
    """Single golden RAG evaluation scenario with ground-truth provenance."""

    scenario_id: str
    domain: str
    query: str
    region: str
    expected_doc_ids: list[str]
    stale_conflicting_doc_id: Optional[str]
    distractor_doc_ids: list[str]
    required_facts: list[str]
    forbidden_stale_facts: list[str]
    should_abstain: bool
    ground_truth_answer: str
    query_entities: list[str] = field(default_factory=list)


def build_enterprise_corpus() -> list[Document]:
    """
    Constructs a 54-document enterprise corpus containing:
      - 24 active 2026 Tier-1/Tier-2 policy & protocol documents
      - 12 superseded 2024 Tier-3 legacy documents with conflicting thresholds
      - 18 topically similar distractor documents
    """
    docs: list[Document] = [
        # =========================================================================
        # DOMAIN 1: SAME-DAY DELIVERY SLA & ROUTING POLICIES (2026 Active)
        # =========================================================================
        Document(
            doc_id="DOC-SSD-2026-01",
            title="Sub-Same-Day (SSD) Sortation Cutoff & Wave Dispatch Policy v4.2",
            content=(
                "Effective February 2026, all Sub-Same-Day (SSD) urban fulfillment nodes "
                "must enforce a strict evening wave order sortation cutoff at 14:30 local "
                "facility time. Orders inducted after 14:30 local time roll to the overnight "
                "04:00 AM window. This 14:30 local cutoff supersedes the legacy 2024 cutoff "
                "schedule to guarantee 99.4% dock-to-door on-time departure."
            ),
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2026-02-15",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SSD-2026-02",
            title="Perishable Cold-Chain Thermal Excursion & Spoilage Protocol",
            content=(
                "For Same-Day grocery and chilled biologics dispatch, insulated totes must "
                "maintain internal temperatures below 4.0°C. A cumulative thermal excursion "
                "above 4.0°C is permitted for a maximum of 22 minutes during last-mile handoff. "
                "Any tote exceeding 22 minutes above 4.0°C must be quarantined and flagged "
                "for automated disposal."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2026-03-10",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SSD-2026-03",
            title="Lithium-Ion Hazmat Air-to-Ground Fallback Routing Standard",
            content=(
                "Packages containing standalone or packed lithium-ion batteries exceeding "
                "100 Wh are prohibited on Same-Day air linehaul feeders and must be rerouted "
                "via dedicated ground sprinter vans with a maximum aggregate energy cap of "
                "300 Wh per parcel. Hazmat parcels above 300 Wh require Class 9 placard "
                "freight carriers."
            ),
            domain="same_day_delivery",
            region="US-WEST",
            effective_date="2026-01-20",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SSD-2026-04",
            title="Dynamic Flex Courier Route Density & Parcel Weight Ceiling",
            content=(
                "Under the 2026 Dynamic Dispatch Algorithm, a 4-hour Same-Day passenger-vehicle "
                "courier block is capped at 42 delivery stops and a maximum single-parcel weight "
                "of 48 lbs. Routes projected to exceed 42 stops must trigger automated route "
                "splitting at the micro-fulfillment staging cage."
            ),
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2026-04-01",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SSD-2026-05",
            title="Micro-Fulfillment Center (MFC) Pick-to-Pack SLA & Buffer Release",
            content=(
                "Automated Micro-Fulfillment Centers (MFCs) operating high-velocity Same-Day "
                "waves must complete pick-to-pack induction within 18 minutes of customer "
                "checkout confirmation. Orders breaching the 18 minutes pick-to-pack SLA "
                "automatically trigger dynamic carrier pre-alert re-slotting."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2026-02-01",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SSD-2026-06",
            title="Severe Weather Lightning Ramp Ground-Stop Safety Directive",
            content=(
                "Open-air Same-Day load-out pads and rooftop drone pads must initiate an "
                "immediate ramp ground-stop whenever cloud-to-ground lightning strikes are "
                "detected within an 8 nautical miles radius. Operations resume only after "
                "30 consecutive minutes with zero strikes inside the 8 nautical miles perimeter."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2026-05-12",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SSD-2026-07",
            title="Smart Locker Dwell Time & Overflow Redirection Policy",
            content=(
                "Same-Day parcels deposited into automated Hub Smart Lockers have a maximum "
                "customer pickup dwell window of 36 hours. If unclaimed after 36 hours, "
                "the courier sweep route retrieves the parcel for restocking at the origin "
                "sub-same-day station."
            ),
            domain="same_day_delivery",
            region="EU-WEST",
            effective_date="2026-03-22",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SSD-2026-08",
            title="Autonomous Sidewalk Delivery Bot Speed & Geofence Governance",
            content=(
                "Six-wheeled autonomous sidewalk delivery units deployed for suburban "
                "Same-Day drops are governed to a maximum operating speed of 12 mph on "
                "designated bike lanes and 4 mph on pedestrian crosswalks, with mandatory "
                "remote tele-operator takeover whenever optical lidar visibility drops below 15 meters."
            ),
            domain="same_day_delivery",
            region="US-WEST",
            effective_date="2026-06-01",
            authority_tier=2,
        ),

        # =========================================================================
        # DOMAIN 1: STALE / SUPERSEDED 2024 DOCUMENTS (Conflicts)
        # =========================================================================
        Document(
            doc_id="DOC-SSD-2024-01",
            title="Sub-Same-Day (SSD) Sortation Cutoff & Wave Dispatch Policy (2024 Legacy)",
            content=(
                "Under the 2024 Sub-Same-Day (SSD) sortation policy, urban fulfillment nodes "
                "maintain an evening wave order sortation cutoff at 16:00 local facility time. "
                "What is the evening wave sortation cutoff time for Sub-Same-Day urban nodes? "
                "The cutoff is 16:00 local time."
            ),
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2024-03-01",
            authority_tier=3,
            superseded_by="DOC-SSD-2026-01",
        ),
        Document(
            doc_id="DOC-SSD-2024-02",
            title="Perishable Cold-Chain Thermal Excursion Protocol (2024 Legacy)",
            content=(
                "In the 2024 cold-chain delivery manual, insulated grocery totes are allowed "
                "a thermal excursion above 4.0°C for up to 45 minutes during last-mile handoff "
                "before quarantine is required."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2024-01-15",
            authority_tier=3,
            superseded_by="DOC-SSD-2026-02",
        ),
        Document(
            doc_id="DOC-SSD-2024-04",
            title="Dynamic Flex Courier Route Density Policy (2024 Legacy)",
            content=(
                "Under the 2024 Flex courier routing rules, a 4-hour Same-Day passenger-vehicle "
                "courier block allows up to 55 delivery stops and a single-parcel weight cap of 70 lbs."
            ),
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2024-05-10",
            authority_tier=3,
            superseded_by="DOC-SSD-2026-04",
        ),
        Document(
            doc_id="DOC-SSD-2024-05",
            title="Micro-Fulfillment Center (MFC) Pick-to-Pack SLA (2024 Legacy)",
            content=(
                "Under the 2024 MFC operations guide, Micro-Fulfillment Centers must complete "
                "pick-to-pack induction within 30 minutes of customer checkout confirmation."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2024-02-19",
            authority_tier=3,
            superseded_by="DOC-SSD-2026-05",
        ),

        # =========================================================================
        # DOMAIN 1: DISTRACTOR DOCUMENTS
        # =========================================================================
        Document(
            doc_id="DOC-SSD-DIST-01",
            title="Standard Freight Linehaul Dock Scheduling & Pallet Cutoff",
            content=(
                "Full-truckload (FTL) regional cross-dock palletized transfers close dock "
                "doors at 21:00 UTC for 3-to-5 day ground economy shipping lanes."
            ),
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2026-01-10",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SSD-DIST-02",
            title="Ambient Pantry Dry Goods Packaging Specification",
            content=(
                "Non-perishable dry pantry items shipped in corrugated boxes are rated for "
                "ambient warehouse storage between 10°C and 32°C for up to 180 days."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2026-01-12",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SSD-DIST-03",
            title="Warehouse Forklift Lead-Acid Battery Charging Bay SOP",
            content=(
                "Industrial reach-truck lead-acid traction batteries rated at 48V / 1200 Ah "
                "must be watered and equalized weekly in ventilated charging bays."
            ),
            domain="same_day_delivery",
            region="US-WEST",
            effective_date="2026-01-18",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SSD-DIST-04",
            title="Commercial Box Truck CDL Driver Hours of Service Log",
            content=(
                "Heavy commercial 26-foot box truck drivers are limited to 11 hours of driving "
                "within a 14-hour on-duty window under federal DOT regulations."
            ),
            domain="same_day_delivery",
            region="US-EAST",
            effective_date="2026-02-05",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SSD-DIST-05",
            title="Fulfillment Center Mezzanine Conveyor Preventive Maintenance",
            content=(
                "High-speed shoe sorter belts require laser alignment inspection every 250 "
                "operating hours with a maximum belt tension variance of 3%."
            ),
            domain="same_day_delivery",
            region="GLOBAL",
            effective_date="2026-02-11",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SSD-DIST-06",
            title="Employee Parking Lot Snow Removal & De-Icing Contract",
            content=(
                "Facility maintenance contractors must spread magnesium chloride de-icer "
                "whenever snow accumulation exceeds 2 inches on employee parking lots."
            ),
            domain="same_day_delivery",
            region="EU-WEST",
            effective_date="2026-01-25",
            authority_tier=2,
        ),

        # =========================================================================
        # DOMAIN 2: SELLING PARTNER SUPPORT & SELLER ASSISTANT (2026 Active)
        # =========================================================================
        Document(
            doc_id="DOC-SPS-2026-01",
            title="FBA Lost-in-Warehouse Reimbursement Valuation Cap Policy v2026",
            content=(
                "Effective January 2026, when inventory is confirmed lost or damaged inside "
                "an FBA fulfillment center, automated Seller Support reimbursement is capped "
                "at $5,000 per single unit without independent third-party appraisal. Items "
                "valued above $5,000 require a certified commercial invoice and underwriting "
                "review within 30 days."
            ),
            domain="seller_support",
            region="US-EAST",
            effective_date="2026-01-15",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SPS-2026-02",
            title="Seller-Fulfilled SAFE-T Claim Filing Window & Evidence Standard",
            content=(
                "Selling partners filing a SAFE-T reimbursement claim for customer-damaged "
                "or materially different merchant-fulfilled returns must submit the claim "
                "within 45 calendar days of the original refund debit date, accompanied by "
                "unboxed photographic evidence of the return tracking label and serial number."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2026-03-01",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SPS-2026-03",
            title="Payment Chargeback Service Fee Waiver Eligibility Policy",
            content=(
                "Selling partners are granted a 100% chargeback processing fee waiver when "
                "the shipment utilizes integrated Buy Shipping labels and exhibits a valid "
                "carrier acceptance tracking scan within 14 calendar days of the order date "
                "plus signature confirmation for orders exceeding $750."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2026-02-20",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SPS-2026-04",
            title="Intellectual Property & Counterfeit Appeal Invoice Verification Rules",
            content=(
                "To reinstate an ASIN suppressed for suspected counterfeit or inauthentic "
                "complaints, the seller appeal must include itemized supplier invoices dated "
                "within a 90 days lookback window prior to the complaint, displaying the "
                "supplier VAT or federal EIN and matching total unit volume."
            ),
            domain="seller_support",
            region="EU-WEST",
            effective_date="2026-04-10",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SPS-2026-05",
            title="FBA Dangerous Goods Hazmat Storage Overage Surcharge Schedule",
            content=(
                "Selling partners exceeding their allocated FBA flammable or aerosol dangerous "
                "goods storage utilization limit incur a monthly hazmat overage surcharge of "
                "$1.85 per cubic foot calculated daily on excess volume above the tier allocation."
            ),
            domain="seller_support",
            region="US-WEST",
            effective_date="2026-02-01",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SPS-2026-06",
            title="High-Value Fine Jewelry Return Gemological Inspection SLA",
            content=(
                "Returned fine jewelry items with a retail price above $1,500 routed to "
                "specialized return centers must undergo gemological verification within "
                "3 business days of dock receipt before restocking or issuing a switcheroo credit."
            ),
            domain="seller_support",
            region="US-EAST",
            effective_date="2026-05-01",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SPS-2026-07",
            title="Brand Registry Pending Trademark Application Enrolment Criteria",
            content=(
                "Brands enrolling via the IP Accelerator or pending trademark pathway must "
                "provide an active USPTO or EUIPO trademark serial number verified within "
                "48 hours of filing and match the exact word mark or design mark specimen "
                "printed on product packaging."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2026-03-15",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-SPS-2026-08",
            title="Seller-Fulfilled Prime (SFP) On-Time Delivery & Trial Thresholds",
            content=(
                "To retain Seller-Fulfilled Prime (SFP) badge eligibility under the 2026 "
                "performance standards, merchants must maintain at least a 97.5% on-time "
                "delivery rate and a 99.0% buy-shipping usage rate evaluated weekly across "
                "all Prime trial and enrolled SKUs."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2026-04-18",
            authority_tier=1,
        ),

        # =========================================================================
        # DOMAIN 2: STALE / SUPERSEDED 2024 DOCUMENTS (Conflicts)
        # =========================================================================
        Document(
            doc_id="DOC-SPS-2024-01",
            title="FBA Lost-in-Warehouse Reimbursement Valuation Cap Policy (2024 Legacy)",
            content=(
                "Under the 2024 FBA reimbursement policy, automated Seller Support "
                "reimbursement for lost-in-warehouse inventory is capped at $2,000 per "
                "single unit without appraisal."
            ),
            domain="seller_support",
            region="US-EAST",
            effective_date="2024-02-01",
            authority_tier=3,
            superseded_by="DOC-SPS-2026-01",
        ),
        Document(
            doc_id="DOC-SPS-2024-02",
            title="Seller-Fulfilled SAFE-T Claim Filing Window (2024 Legacy)",
            content=(
                "Under the 2024 SAFE-T reimbursement policy, selling partners have up to "
                "60 calendar days from the refund debit date to submit a SAFE-T claim."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2024-04-01",
            authority_tier=3,
            superseded_by="DOC-SPS-2026-02",
        ),
        Document(
            doc_id="DOC-SPS-2024-04",
            title="Counterfeit Appeal Invoice Verification Rules (2024 Legacy)",
            content=(
                "Under the 2024 counterfeit appeal rules, sellers may submit supplier "
                "invoices dated within a 365 days lookback window prior to the complaint."
            ),
            domain="seller_support",
            region="EU-WEST",
            effective_date="2024-01-10",
            authority_tier=3,
            superseded_by="DOC-SPS-2026-04",
        ),
        Document(
            doc_id="DOC-SPS-2024-08",
            title="Seller-Fulfilled Prime (SFP) On-Time Delivery Threshold (2024 Legacy)",
            content=(
                "Under the 2024 Seller-Fulfilled Prime (SFP) requirements, merchants must "
                "maintain at least a 93.5% on-time delivery rate."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2024-06-01",
            authority_tier=3,
            superseded_by="DOC-SPS-2026-08",
        ),

        # =========================================================================
        # DOMAIN 2: DISTRACTOR DOCUMENTS
        # =========================================================================
        Document(
            doc_id="DOC-SPS-DIST-01",
            title="FBA Monthly Long-Term Aged Inventory Surcharge Tier Table",
            content=(
                "Inventory stored in fulfillment centers for 271 to 365 days incurs an "
                "aged inventory surcharge of $3.90 per cubic foot or $0.15 per unit."
            ),
            domain="seller_support",
            region="US-EAST",
            effective_date="2026-01-05",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SPS-DIST-02",
            title="Sponsored Products Cost-Per-Click Daily Budget Billing Rules",
            content=(
                "Advertising campaign daily budgets may pace up to 25% above the daily "
                "target on high-traffic days while capping monthly spend at 30.4x daily budget."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2026-01-20",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SPS-DIST-03",
            title="A-to-Z Guarantee Customer Claim Appeal Response Timeline",
            content=(
                "When a buyer opens an A-to-Z Guarantee claim, the seller must respond to "
                "investigator information requests within 72 hours."
            ),
            domain="seller_support",
            region="GLOBAL",
            effective_date="2026-02-12",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SPS-DIST-04",
            title="Product Detail Page Bullet Point & Title Style Guidelines",
            content=(
                "Product titles must not exceed 200 characters including spaces and may "
                "not contain promotional phrases such as 'free shipping' or 'best seller'."
            ),
            domain="seller_support",
            region="EU-WEST",
            effective_date="2026-01-19",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SPS-DIST-05",
            title="Amazon Vine Customer Review Enrolment Fee Schedule",
            content=(
                "Enrolling 11 to 30 units of a parent ASIN into the Vine review program "
                "incurs a flat $200 enrolment fee billed after the first published review."
            ),
            domain="seller_support",
            region="US-WEST",
            effective_date="2026-02-08",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-SPS-DIST-06",
            title="Apparel Size Chart & Return Reason Analytics Dashboard",
            content=(
                "Apparel listings with more than 12% fit-related returns receive an automated "
                "size chart standardization notification in Seller Central."
            ),
            domain="seller_support",
            region="US-EAST",
            effective_date="2026-03-04",
            authority_tier=2,
        ),

        # =========================================================================
        # DOMAIN 3: CLINICAL & REGULATORY PROTOCOL COMPLIANCE (2026 Active)
        # =========================================================================
        Document(
            doc_id="DOC-CLN-2026-01",
            title="Oncology Phase III Protocol ONC-309 Amendment 4: ANC Dosing Hold",
            content=(
                "Per 2026 Protocol Amendment 4 for trial ONC-309, investigative cycle day-1 "
                "dosing must be immediately held whenever a patient's Absolute Neutrophil "
                "Count (ANC) falls below 1,000 cells/mm3 (Grade 3 neutropenia). Dosing may "
                "resume only after ANC recovers to >= 1,500 cells/mm3."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-03-05",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-02",
            title="FDA IND Safety Reporting SOP: Fatal or Life-Threatening Unexpected SAEs",
            content=(
                "Under 21 CFR 312.32 and the 2026 Pharmacovigilance SOP, any unexpected "
                "fatal or life-threatening Suspected Adverse Reaction (SUSAR/SAE) must be "
                "transmitted to the FDA and central IRB within 7 calendar days of initial "
                "sponsor awareness, followed by a complete follow-up report within 8 additional days."
            ),
            domain="clinical_protocol",
            region="US-EAST",
            effective_date="2026-01-10",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-03",
            title="Cardiac Safety Monitoring: QTcF Prolongation Stopping Rule",
            content=(
                "Subjects enrolled in cardiovascular or oncology cohorts must permanently "
                "discontinue study drug if triplicate 12-lead ECG confirms a Fridericia-corrected "
                "QT interval (QTcF) exceeding 500 ms or an absolute increase of > 60 ms "
                "from pre-dose baseline."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-02-18",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-04",
            title="Monoclonal Antibody Investigational Product Cold-Storage Stability Protocol",
            content=(
                "Lyophilized and liquid monoclonal antibody (mAb) vials must be stored at "
                "2°C to 8°C in continuous-monitored pharmacy refrigerators. A single room-temperature "
                "excursion up to 25°C is permissible for a maximum of 4 hours prior to dilution; "
                "vials exposed beyond 4 hours at 25°C must be destroyed."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-04-02",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-05",
            title="Informed Consent Form (ICF) Re-Consenting Timeline After Safety Amendment",
            content=(
                "Whenever an urgent safety amendment updates the Informed Consent Form (ICF) "
                "risk profile, active trial participants must sign the revised IRB-approved "
                "ICF within 14 calendar days of site activation or at their next scheduled "
                "clinic visit, whichever occurs first."
            ),
            domain="clinical_protocol",
            region="US-WEST",
            effective_date="2026-02-28",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-06",
            title="Hepatic Toxicity Dose Modification Guidelines (AST/ALT Grade 3)",
            content=(
                "For subjects experiencing Grade 3 hepatic transaminase elevation (AST or ALT "
                "> 5.0 to 20.0 x ULN without bilirubin elevation), withhold study drug until "
                "recovery to Grade <= 1, then resume treatment at a mandatory 50% dose reduction."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-03-19",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-07",
            title="Pediatric Adolescent Cohort Weight-Based Exclusion Criteria",
            content=(
                "In adolescent expansion Cohort B (ages 12 to 17), subjects with a screening "
                "body weight below 35 kg are excluded from fixed-dose tablet administration "
                "to prevent supratherapeutic Cmax exposure."
            ),
            domain="clinical_protocol",
            region="EU-WEST",
            effective_date="2026-05-05",
            authority_tier=1,
        ),
        Document(
            doc_id="DOC-CLN-2026-08",
            title="Electronic Clinical Outcome Assessment (eCOA) Run-In Compliance Floor",
            content=(
                "Prior to randomization at Visit 2, subjects must demonstrate an electronic "
                "patient diary (eCOA) completion compliance rate of at least 85% across the "
                "14-day screening run-in period. Subjects below 85% eCOA compliance are "
                "screen-failed."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-04-14",
            authority_tier=2,
        ),

        # =========================================================================
        # DOMAIN 3: STALE / SUPERSEDED 2024 DOCUMENTS (Conflicts)
        # =========================================================================
        Document(
            doc_id="DOC-CLN-2024-01",
            title="Oncology Phase III Protocol ONC-309 Original v1: ANC Dosing Hold (2024)",
            content=(
                "Under the 2024 Original Protocol v1 for trial ONC-309, cycle day-1 dosing "
                "is held whenever Absolute Neutrophil Count (ANC) falls below 1,500 cells/mm3."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2024-01-20",
            authority_tier=3,
            superseded_by="DOC-CLN-2026-01",
        ),
        Document(
            doc_id="DOC-CLN-2024-04",
            title="Monoclonal Antibody Cold-Storage Excursion Protocol (2024 Legacy)",
            content=(
                "Under the 2024 pharmacy manual, monoclonal antibody (mAb) vials may tolerate "
                "a room-temperature excursion up to 25°C for up to 12 hours prior to dilution."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2024-03-11",
            authority_tier=3,
            superseded_by="DOC-CLN-2026-04",
        ),
        Document(
            doc_id="DOC-CLN-2024-05",
            title="Informed Consent Form (ICF) Re-Consenting Timeline (2024 Legacy)",
            content=(
                "Under the 2024 site operations guideline, active trial participants must sign "
                "a revised Informed Consent Form (ICF) within 45 calendar days of site activation."
            ),
            domain="clinical_protocol",
            region="US-WEST",
            effective_date="2024-05-01",
            authority_tier=3,
            superseded_by="DOC-CLN-2026-05",
        ),
        Document(
            doc_id="DOC-CLN-2024-08",
            title="Electronic Diary (eCOA) Run-In Compliance Floor (2024 Legacy)",
            content=(
                "Under the 2024 protocol version, subjects must achieve at least 70% eCOA "
                "diary compliance during the screening run-in period to qualify for randomization."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2024-02-15",
            authority_tier=3,
            superseded_by="DOC-CLN-2026-08",
        ),

        # =========================================================================
        # DOMAIN 3: DISTRACTOR DOCUMENTS
        # =========================================================================
        Document(
            doc_id="DOC-CLN-DIST-01",
            title="Routine Hematology Venipuncture Tube Labeling & Centrifuge SOP",
            content=(
                "Whole blood EDTA lavender-top tubes must be inverted 8 times immediately "
                "after collection and processed within 2 hours of phlebotomy."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-01-08",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-CLN-DIST-02",
            title="Annual IND Development Safety Update Report (DSUR) Formatting Guide",
            content=(
                "Periodic annual DSUR submissions must be compiled within 60 days of the "
                "Development International Birth Date (DIBD) in eCTD Module 5 format."
            ),
            domain="clinical_protocol",
            region="US-EAST",
            effective_date="2026-01-16",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-CLN-DIST-03",
            title="Ambulatory Blood Pressure Cuff Calibration & Holter Lead Placement",
            content=(
                "Oscillometric blood pressure cuffs must be calibrated semi-annually against "
                "a NIST-traceable mercury manometer within +/- 2 mmHg."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-02-02",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-CLN-DIST-04",
            title="Bioanalytical PK Plasma Sample Dry-Ice Shipment Manifest",
            content=(
                "Frozen pharmacokinetic plasma aliquots must be packed with a minimum of "
                "15 kg of pelleted dry ice (-78.5°C) for central lab courier transit."
            ),
            domain="clinical_protocol",
            region="GLOBAL",
            effective_date="2026-02-14",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-CLN-DIST-05",
            title="Clinical Investigator Financial Disclosure Form FDA 3455 Instructions",
            content=(
                "Principal and sub-investigators must disclose significant equity interests "
                "exceeding $50,000 in the sponsor organization prior to site initiation."
            ),
            domain="clinical_protocol",
            region="US-WEST",
            effective_date="2026-01-29",
            authority_tier=2,
        ),
        Document(
            doc_id="DOC-CLN-DIST-06",
            title="Site Pharmacy Double-Blind Interactive Response Technology (IRT) Login",
            content=(
                "Unblinded site pharmacists receive two-factor IRT credentials within 24 hours "
                "of completing GCP training module certification."
            ),
            domain="clinical_protocol",
            region="EU-WEST",
            effective_date="2026-03-01",
            authority_tier=2,
        ),
    ]
    return docs


def build_golden_scenarios() -> list[EvaluationScenario]:
    """
    Constructs 30 golden evaluation scenarios across the 3 target domains:
      - 24 answerable scenarios (12 of which have direct 2024 vs 2026 supersession conflicts)
      - 6 unanswerable scenarios where the corpus lacks the requested policy/threshold
        and the system MUST abstain rather than guess from distractors.
    """
    return [
        # =========================================================================
        # DOMAIN 1: SAME-DAY DELIVERY (10 Scenarios: 8 Answerable, 2 Abstention)
        # =========================================================================
        EvaluationScenario(
            scenario_id="SCN-SSD-01",
            domain="same_day_delivery",
            query="What is the current evening wave order sortation cutoff time for Sub-Same-Day (SSD) urban fulfillment nodes?",
            region="US-EAST",
            expected_doc_ids=["DOC-SSD-2026-01"],
            stale_conflicting_doc_id="DOC-SSD-2024-01",
            distractor_doc_ids=["DOC-SSD-DIST-01", "DOC-SSD-DIST-05"],
            required_facts=["14:30"],
            forbidden_stale_facts=["16:00"],
            should_abstain=False,
            ground_truth_answer=(
                "Per current 2026 policy [DOC-SSD-2026-01], the evening wave order sortation "
                "cutoff for Sub-Same-Day (SSD) urban fulfillment nodes is 14:30 local facility time "
                "(superseding the legacy 2024 16:00 cutoff)."
            ),
            query_entities=["sortation", "cutoff", "sub-same-day", "ssd"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-02",
            domain="same_day_delivery",
            query="How many cumulative minutes can a Same-Day perishable cold-chain tote exceed 4.0°C during last-mile handoff before quarantine?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SSD-2026-02"],
            stale_conflicting_doc_id="DOC-SSD-2024-02",
            distractor_doc_ids=["DOC-SSD-DIST-02"],
            required_facts=["22 minutes", "4.0°C"],
            forbidden_stale_facts=["45 minutes"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-02], a perishable cold-chain tote may experience a cumulative "
                "thermal excursion above 4.0°C for a maximum of 22 minutes during last-mile handoff "
                "before mandatory quarantine and disposal."
            ),
            query_entities=["cold-chain", "excursion", "4.0", "quarantine"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-03",
            domain="same_day_delivery",
            query="What is the maximum aggregate energy cap in watt-hours (Wh) per parcel for lithium-ion batteries rerouted to ground sprinter vans?",
            region="US-WEST",
            expected_doc_ids=["DOC-SSD-2026-03"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SSD-DIST-03"],
            required_facts=["300 Wh"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-03], lithium-ion battery parcels rerouted to ground sprinter "
                "vans have a maximum aggregate energy cap of 300 Wh per parcel."
            ),
            query_entities=["lithium-ion", "wh", "sprinter", "hazmat"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-04",
            domain="same_day_delivery",
            query="Under the current Dynamic Dispatch Algorithm, what is the maximum number of delivery stops and single-parcel weight limit for a 4-hour courier block?",
            region="US-EAST",
            expected_doc_ids=["DOC-SSD-2026-04"],
            stale_conflicting_doc_id="DOC-SSD-2024-04",
            distractor_doc_ids=["DOC-SSD-DIST-04"],
            required_facts=["42", "48 lbs"],
            forbidden_stale_facts=["55", "70 lbs"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-04], a 4-hour Same-Day passenger-vehicle courier block is "
                "capped at 42 delivery stops and a maximum single-parcel weight of 48 lbs."
            ),
            query_entities=["stops", "courier", "4-hour", "weight"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-05",
            domain="same_day_delivery",
            query="Within how many minutes of checkout confirmation must an automated Micro-Fulfillment Center (MFC) complete pick-to-pack induction?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SSD-2026-05"],
            stale_conflicting_doc_id="DOC-SSD-2024-05",
            distractor_doc_ids=["DOC-SSD-DIST-05"],
            required_facts=["18 minutes"],
            forbidden_stale_facts=["30 minutes"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-05], automated Micro-Fulfillment Centers (MFCs) must complete "
                "pick-to-pack induction within 18 minutes of customer checkout confirmation."
            ),
            query_entities=["micro-fulfillment", "mfc", "pick-to-pack", "induction"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-06",
            domain="same_day_delivery",
            query="At what lightning strike radius in nautical miles must open-air Same-Day load-out pads initiate a ramp ground-stop?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SSD-2026-06"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SSD-DIST-06"],
            required_facts=["8 nautical miles", "30"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-06], open-air Same-Day load-out pads must initiate a ramp "
                "ground-stop whenever lightning is detected within an 8 nautical miles radius, "
                "resuming after 30 consecutive strike-free minutes."
            ),
            query_entities=["lightning", "nautical", "ground-stop", "ramp"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-07",
            domain="same_day_delivery",
            query="What is the maximum customer pickup dwell time window in hours for Same-Day parcels inside Hub Smart Lockers before courier retrieval?",
            region="EU-WEST",
            expected_doc_ids=["DOC-SSD-2026-07"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SSD-DIST-01"],
            required_facts=["36 hours"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-07], Same-Day parcels in Hub Smart Lockers have a maximum "
                "customer pickup dwell window of 36 hours before courier retrieval."
            ),
            query_entities=["locker", "dwell", "hours", "pickup"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-08",
            domain="same_day_delivery",
            query="What is the maximum operating speed on bike lanes and pedestrian crosswalks for autonomous sidewalk delivery bots?",
            region="US-WEST",
            expected_doc_ids=["DOC-SSD-2026-08"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SSD-DIST-04"],
            required_facts=["12 mph", "4 mph"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SSD-2026-08], autonomous sidewalk delivery bots are limited to "
                "12 mph on bike lanes and 4 mph on pedestrian crosswalks."
            ),
            query_entities=["autonomous", "sidewalk", "mph", "crosswalks"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-09",
            domain="same_day_delivery",
            query="What is the maximum payload weight and wind-shear knot limit for maritime hovercraft Same-Day island delivery in APAC-OKINAWA?",
            region="APAC-OKINAWA",
            expected_doc_ids=[],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SSD-2026-06", "DOC-SSD-2026-04"],
            required_facts=[],
            forbidden_stale_facts=[],
            should_abstain=True,
            ground_truth_answer=(
                "ABSTAIN: The corpus contains no policy governing maritime hovercraft island "
                "delivery payload or wind-shear limits in APAC-OKINAWA."
            ),
            query_entities=["hovercraft", "maritime", "okinawa", "wind-shear"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SSD-10",
            domain="same_day_delivery",
            query="What is the customs de-minimis duty waiver threshold for cross-border helicopter Same-Day delivery into Switzerland?",
            region="EU-WEST",
            expected_doc_ids=[],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SSD-2026-07", "DOC-SSD-DIST-01"],
            required_facts=[],
            forbidden_stale_facts=[],
            should_abstain=True,
            ground_truth_answer=(
                "ABSTAIN: No policy document in the knowledge base specifies cross-border "
                "helicopter customs de-minimis thresholds for Switzerland."
            ),
            query_entities=["helicopter", "customs", "switzerland", "de-minimis"],
        ),

        # =========================================================================
        # DOMAIN 2: SELLING PARTNER SUPPORT (10 Scenarios: 8 Answerable, 2 Abstention)
        # =========================================================================
        EvaluationScenario(
            scenario_id="SCN-SPS-11",
            domain="seller_support",
            query="What is the automated Seller Support reimbursement valuation cap per single unit for FBA lost-in-warehouse inventory without independent appraisal?",
            region="US-EAST",
            expected_doc_ids=["DOC-SPS-2026-01"],
            stale_conflicting_doc_id="DOC-SPS-2024-01",
            distractor_doc_ids=["DOC-SPS-DIST-01"],
            required_facts=["$5,000"],
            forbidden_stale_facts=["$2,000"],
            should_abstain=False,
            ground_truth_answer=(
                "Per 2026 policy [DOC-SPS-2026-01], automated FBA lost-in-warehouse "
                "reimbursement is capped at $5,000 per single unit without independent appraisal."
            ),
            query_entities=["fba", "lost-in-warehouse", "reimbursement", "cap"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-12",
            domain="seller_support",
            query="Within how many calendar days of the original refund debit date must a selling partner file a SAFE-T reimbursement claim?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SPS-2026-02"],
            stale_conflicting_doc_id="DOC-SPS-2024-02",
            distractor_doc_ids=["DOC-SPS-DIST-03"],
            required_facts=["45 calendar days"],
            forbidden_stale_facts=["60 calendar days"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-02], selling partners must submit a SAFE-T reimbursement "
                "claim within 45 calendar days of the original refund debit date."
            ),
            query_entities=["safe-t", "claim", "calendar", "refund"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-13",
            domain="seller_support",
            query="What carrier tracking scan window and signature threshold are required to qualify for a 100% payment chargeback fee waiver?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SPS-2026-03"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SPS-DIST-03"],
            required_facts=["14 calendar days", "$750"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-03], a 100% chargeback fee waiver requires Buy Shipping "
                "with a carrier acceptance scan within 14 calendar days of the order date "
                "and signature confirmation for orders above $750."
            ),
            query_entities=["chargeback", "waiver", "signature", "scan"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-14",
            domain="seller_support",
            query="What is the required invoice lookback window in days when appealing an ASIN suppression for suspected counterfeit complaints?",
            region="EU-WEST",
            expected_doc_ids=["DOC-SPS-2026-04"],
            stale_conflicting_doc_id="DOC-SPS-2024-04",
            distractor_doc_ids=["DOC-SPS-DIST-04"],
            required_facts=["90 days"],
            forbidden_stale_facts=["365 days"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-04], counterfeit appeal invoices must be dated within a "
                "90 days lookback window prior to the complaint and include supplier VAT/EIN."
            ),
            query_entities=["counterfeit", "invoices", "lookback", "asin"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-15",
            domain="seller_support",
            query="What is the monthly FBA dangerous goods hazmat storage overage surcharge per cubic foot when exceeding allocated volume?",
            region="US-WEST",
            expected_doc_ids=["DOC-SPS-2026-05"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SPS-DIST-01"],
            required_facts=["$1.85"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-05], the FBA dangerous goods hazmat storage overage "
                "surcharge is $1.85 per cubic foot on excess volume."
            ),
            query_entities=["hazmat", "overage", "surcharge", "cubic"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-16",
            domain="seller_support",
            query="Within how many business days of dock receipt must returned fine jewelry above $1,500 undergo gemological inspection?",
            region="US-EAST",
            expected_doc_ids=["DOC-SPS-2026-06"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SPS-DIST-06"],
            required_facts=["3 business days"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-06], returned fine jewelry priced above $1,500 must "
                "undergo gemological verification within 3 business days of dock receipt."
            ),
            query_entities=["jewelry", "gemological", "business", "dock"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-17",
            domain="seller_support",
            query="What trademark verification window and offices are required for Brand Registry enrolment with a pending trademark application?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SPS-2026-07"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SPS-DIST-04"],
            required_facts=["48 hours", "USPTO", "EUIPO"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-07], Brand Registry pending trademark enrolment requires "
                "an active USPTO or EUIPO serial number verified within 48 hours of filing."
            ),
            query_entities=["trademark", "brand", "registry", "uspto"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-18",
            domain="seller_support",
            query="What minimum on-time delivery percentage must a merchant maintain to keep Seller-Fulfilled Prime (SFP) badge eligibility in 2026?",
            region="GLOBAL",
            expected_doc_ids=["DOC-SPS-2026-08"],
            stale_conflicting_doc_id="DOC-SPS-2024-08",
            distractor_doc_ids=["DOC-SPS-DIST-02"],
            required_facts=["97.5%"],
            forbidden_stale_facts=["93.5%"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-SPS-2026-08], merchants must maintain at least a 97.5% on-time "
                "delivery rate (and 99.0% buy-shipping usage) to retain Seller-Fulfilled Prime (SFP)."
            ),
            query_entities=["seller-fulfilled", "sfp", "on-time", "prime"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-19",
            domain="seller_support",
            query="What is the cryptocurrency escrow settlement fee percentage for NFT digital collectibles sold on the Brazil marketplace?",
            region="SA-EAST",
            expected_doc_ids=[],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SPS-2026-03", "DOC-SPS-DIST-02"],
            required_facts=[],
            forbidden_stale_facts=[],
            should_abstain=True,
            ground_truth_answer=(
                "ABSTAIN: The corpus contains no policy regarding cryptocurrency escrow or "
                "NFT digital collectible settlement fees on the Brazil marketplace."
            ),
            query_entities=["cryptocurrency", "nft", "escrow", "brazil"],
        ),
        EvaluationScenario(
            scenario_id="SCN-SPS-20",
            domain="seller_support",
            query="What is the maximum livestock quarantine reimbursement rate for live equestrian sales through FBA?",
            region="US-EAST",
            expected_doc_ids=[],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-SPS-2026-01", "DOC-SPS-2026-05"],
            required_facts=[],
            forbidden_stale_facts=[],
            should_abstain=True,
            ground_truth_answer=(
                "ABSTAIN: No policy in the corpus covers live equestrian or livestock sales "
                "reimbursement in FBA."
            ),
            query_entities=["livestock", "equestrian", "quarantine"],
        ),

        # =========================================================================
        # DOMAIN 3: CLINICAL PROTOCOL COMPLIANCE (10 Scenarios: 8 Answerable, 2 Abstention)
        # =========================================================================
        EvaluationScenario(
            scenario_id="SCN-CLN-21",
            domain="clinical_protocol",
            query="In Oncology Phase III trial ONC-309, below what Absolute Neutrophil Count (ANC) threshold must cycle day-1 dosing be held?",
            region="GLOBAL",
            expected_doc_ids=["DOC-CLN-2026-01"],
            stale_conflicting_doc_id="DOC-CLN-2024-01",
            distractor_doc_ids=["DOC-CLN-DIST-01"],
            required_facts=["1,000 cells/mm3"],
            forbidden_stale_facts=["below 1,500 cells/mm3"],
            should_abstain=False,
            ground_truth_answer=(
                "Per 2026 Amendment 4 [DOC-CLN-2026-01], cycle day-1 dosing in trial ONC-309 "
                "must be held when ANC falls below 1,000 cells/mm3."
            ),
            query_entities=["anc", "neutrophil", "onc-309", "dosing"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-22",
            domain="clinical_protocol",
            query="Within how many calendar days of initial sponsor awareness must an unexpected fatal or life-threatening SAE be reported to the FDA?",
            region="US-EAST",
            expected_doc_ids=["DOC-CLN-2026-02"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-CLN-DIST-02"],
            required_facts=["7 calendar days"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-02], unexpected fatal or life-threatening SAEs must be "
                "reported to the FDA and central IRB within 7 calendar days of initial awareness."
            ),
            query_entities=["sae", "fatal", "life-threatening", "fda"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-23",
            domain="clinical_protocol",
            query="What QTcF interval threshold in milliseconds or baseline delta requires permanent discontinuation of study drug?",
            region="GLOBAL",
            expected_doc_ids=["DOC-CLN-2026-03"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-CLN-DIST-03"],
            required_facts=["500 ms", "60 ms"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-03], study drug must be permanently discontinued if triplicate "
                "ECG confirms QTcF > 500 ms or an increase of > 60 ms from baseline."
            ),
            query_entities=["qtcf", "ecg", "500", "discontinue"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-24",
            domain="clinical_protocol",
            query="What is the maximum permissible room-temperature excursion duration at 25°C for monoclonal antibody (mAb) vials prior to dilution?",
            region="GLOBAL",
            expected_doc_ids=["DOC-CLN-2026-04"],
            stale_conflicting_doc_id="DOC-CLN-2024-04",
            distractor_doc_ids=["DOC-CLN-DIST-04"],
            required_facts=["4 hours", "25°C"],
            forbidden_stale_facts=["12 hours"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-04], monoclonal antibody (mAb) vials stored at 2°C to 8°C "
                "may undergo a single excursion up to 25°C for a maximum of 4 hours prior to dilution."
            ),
            query_entities=["monoclonal", "mab", "25°c", "dilution"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-25",
            domain="clinical_protocol",
            query="Within how many calendar days of site activation must active trial participants sign a revised Informed Consent Form (ICF) after a safety amendment?",
            region="US-WEST",
            expected_doc_ids=["DOC-CLN-2026-05"],
            stale_conflicting_doc_id="DOC-CLN-2024-05",
            distractor_doc_ids=["DOC-CLN-DIST-05"],
            required_facts=["14 calendar days"],
            forbidden_stale_facts=["45 calendar days"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-05], active participants must sign the revised IRB-approved "
                "ICF within 14 calendar days of site activation or at their next scheduled visit."
            ),
            query_entities=["icf", "informed", "consent", "re-consenting"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-26",
            domain="clinical_protocol",
            query="What dose reduction percentage is required when resuming study drug after recovery from Grade 3 hepatic AST/ALT elevation?",
            region="GLOBAL",
            expected_doc_ids=["DOC-CLN-2026-06"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-CLN-DIST-01"],
            required_facts=["50%"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-06], following recovery of Grade 3 AST/ALT elevation to "
                "Grade <= 1, treatment resumes at a mandatory 50% dose reduction."
            ),
            query_entities=["hepatic", "ast", "alt", "grade", "reduction"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-27",
            domain="clinical_protocol",
            query="Below what screening body weight in kilograms are adolescent Cohort B subjects excluded from fixed-dose tablet administration?",
            region="EU-WEST",
            expected_doc_ids=["DOC-CLN-2026-07"],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-CLN-DIST-06"],
            required_facts=["35 kg"],
            forbidden_stale_facts=[],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-07], adolescent Cohort B subjects (ages 12 to 17) with a "
                "screening body weight below 35 kg are excluded from fixed-dose tablet dosing."
            ),
            query_entities=["adolescent", "weight", "kg", "cohort"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-28",
            domain="clinical_protocol",
            query="What minimum eCOA electronic patient diary compliance rate across the 14-day run-in period is required for randomization?",
            region="GLOBAL",
            expected_doc_ids=["DOC-CLN-2026-08"],
            stale_conflicting_doc_id="DOC-CLN-2024-08",
            distractor_doc_ids=["DOC-CLN-DIST-06"],
            required_facts=["85%"],
            forbidden_stale_facts=["70%"],
            should_abstain=False,
            ground_truth_answer=(
                "Per [DOC-CLN-2026-08], subjects must achieve at least an 85% eCOA diary "
                "completion compliance rate across the 14-day screening run-in period."
            ),
            query_entities=["ecoa", "diary", "compliance", "randomization"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-29",
            domain="clinical_protocol",
            query="What is the intrathecal radiation dosimetry limit in Gray (Gy) for proton beam therapy in glioblastoma Trial GBM-992?",
            region="GLOBAL",
            expected_doc_ids=[],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-CLN-2026-01", "DOC-CLN-2026-03"],
            required_facts=[],
            forbidden_stale_facts=[],
            should_abstain=True,
            ground_truth_answer=(
                "ABSTAIN: The corpus contains no protocol documents for glioblastoma Trial "
                "GBM-992 or intrathecal proton beam radiation dosimetry limits."
            ),
            query_entities=["glioblastoma", "gbm-992", "proton", "dosimetry"],
        ),
        EvaluationScenario(
            scenario_id="SCN-CLN-30",
            domain="clinical_protocol",
            query="What is the neonatal umbilical cord stem cell cryopreservation DMSO concentration cap in Trial NEO-104?",
            region="US-EAST",
            expected_doc_ids=[],
            stale_conflicting_doc_id=None,
            distractor_doc_ids=["DOC-CLN-2026-04", "DOC-CLN-DIST-04"],
            required_facts=[],
            forbidden_stale_facts=[],
            should_abstain=True,
            ground_truth_answer=(
                "ABSTAIN: No protocol in the knowledge base covers Trial NEO-104 or neonatal "
                "umbilical cord stem cell DMSO cryopreservation thresholds."
            ),
            query_entities=["neonatal", "umbilical", "dmso", "neo-104"],
        ),
    ]
