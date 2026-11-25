/* VeriRAG 3-Domain Grounded Fact-Checking Studio & Shadow Router
 * Stateless in-memory & URL-hash state (strictly no web storage APIs).
 */
(function () {
  "use strict";

  const root = document.documentElement;
  root.classList.add("js");

  const themeBtn = document.querySelector(".theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      history.replaceState(null, "", next === "dark" ? "#theme=dark" : location.pathname);
    });
  }

  // Sidebar progress & active section indicator
  const side = document.querySelector(".side");
  const postBody = document.querySelector(".post-body");
  const sideLinks = side ? [...side.querySelectorAll(".side-list a")] : [];
  const sideTargets = sideLinks.map((a) => document.getElementById(a.hash.slice(1)));
  const sideInd = side && side.querySelector(".side-ind");
  const sideBar = side && side.querySelector(".side-bar i");
  const sidePct = side && side.querySelector(".side-pct");
  const toTop = document.querySelector(".totop");
  const ring = toTop && toTop.querySelector(".ring");
  const RING = 2 * Math.PI * 21;
  if (ring) ring.style.strokeDasharray = String(RING);

  function updateScrollChrome() {
    const y = window.scrollY;
    const max = Math.max(1, root.scrollHeight - window.innerHeight);
    const whole = Math.min(1, Math.max(0, y / max));
    if (toTop && ring) {
      toTop.classList.toggle("show", y > 500);
      ring.style.strokeDashoffset = String(RING * (1 - whole));
    }
    if (side && postBody && sideBar && sidePct) {
      const r = postBody.getBoundingClientRect();
      const read = Math.min(1, Math.max(0, (window.innerHeight * 0.45 - r.top) / Math.max(1, r.height)));
      sideBar.style.transform = `scaleX(${read})`;
      sidePct.textContent = Math.round(read * 100) + "%";
      let active = sideLinks[0] || null;
      sideTargets.forEach((t, i) => {
        if (t && t.getBoundingClientRect().top <= 140) active = sideLinks[i];
      });
      sideLinks.forEach((a) => a.classList.toggle("on", a === active));
      if (active && sideInd) {
        const listRect = side.querySelector(".side-list").getBoundingClientRect();
        const aRect = active.getBoundingClientRect();
        sideInd.style.transform = `translateY(${aRect.top - listRect.top}px)`;
        sideInd.style.height = aRect.height + "px";
      }
    }
  }
  window.addEventListener("scroll", updateScrollChrome, { passive: true });
  window.addEventListener("resize", updateScrollChrome);

  const state = {
    domains: [],
    activeDomainId: "rufus_catalog",
    activePresetId: "rufus-powerbank",
    region: "GLOBAL",
    activeTab: "grounding",
    currentResult: null,
    benchmarkSummary: null,
  };

  function escapeHtml(val) {
    return String(val == null ? "" : val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function highlightCitations(text) {
    return escapeHtml(text).replace(
      /\[(DOC-[A-Z0-9-]+)\]/g,
      '<span class="doc-cite-pill">[$1]</span>'
    );
  }

  function renderDomainsAndPresets() {
    const domWrap = document.getElementById("domainTabs");
    domWrap.innerHTML = state.domains
      .map((d) => {
        const active = d.id === state.activeDomainId;
        return `<button type="button" class="dom-tab ${active ? "active" : ""}" data-did="${escapeHtml(d.id)}">
          <span class="dom-badge">${escapeHtml(d.badge)}</span>
          <span class="dom-title">${escapeHtml(d.short_title)}</span>
        </button>`;
      })
      .join("");

    domWrap.querySelectorAll(".dom-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeDomainId = btn.getAttribute("data-did");
        const dom = state.domains.find((x) => x.id === state.activeDomainId);
        if (dom && dom.presets && dom.presets[0]) {
          const p0 = dom.presets[0];
          state.activePresetId = p0.id;
          state.region = p0.region || "GLOBAL";
          document.getElementById("regionSelect").value = state.region;
          document.getElementById("verifyQueryInput").value = p0.query;
          renderDomainsAndPresets();
          executeVerify(true);
        }
      });
    });

    const curDom = state.domains.find((x) => x.id === state.activeDomainId);
    const presetWrap = document.getElementById("presetButtons");
    if (curDom) {
      presetWrap.innerHTML = (curDom.presets || [])
        .map((p) => {
          const on = p.id === state.activePresetId;
          return `<button type="button" class="preset-btn ${on ? "active" : ""}" data-pid="${escapeHtml(p.id)}">${escapeHtml(p.label)}</button>`;
        })
        .join("");
      presetWrap.querySelectorAll(".preset-btn").forEach((b) => {
        b.addEventListener("click", () => {
          const pid = b.getAttribute("data-pid");
          const pObj = curDom.presets.find((x) => x.id === pid);
          if (!pObj) return;
          state.activePresetId = pObj.id;
          state.region = pObj.region || "GLOBAL";
          document.getElementById("regionSelect").value = state.region;
          document.getElementById("verifyQueryInput").value = pObj.query;
          renderDomainsAndPresets();
          executeVerify(true);
        });
      });
    }
  }

  function renderVerificationResult(res) {
    if (!res) return;
    state.currentResult = res;
    const ans = res.answer;
    const llm = ans.llm_telemetry || {};

    document.getElementById("liveEngineBadge").innerHTML = `
      <span class="tag cobalt">${escapeHtml(llm.engine || "aws-bedrock (us.amazon.nova-lite-v1:0)")}</span>
      <span class="mono-kicker" style="margin-left:6px;">${res.wall_time_ms} ms</span>
    `;

    // 1. Answer / Safe Abstention Banner
    const banner = document.getElementById("answerBanner");
    banner.className = "answer-panel" + (ans.abstained ? " abstained" : "");
    const statusTag = ans.abstained
      ? `<span class="tag amber">🛡️ SAFE ABSTENTION ENFORCED · ZERO HALLUCINATION</span>`
      : `<span class="tag ok">✓ VERIFIED GROUNDED SYNTHESIS · ${escapeHtml((ans.cited_doc_ids || []).join(", "))}</span>`;
    const droppedInfo =
      ans.superseded_docs_dropped && ans.superseded_docs_dropped.length
        ? `<span class="tag bad">EXCLUDED STALE: ${escapeHtml(ans.superseded_docs_dropped.join(", "))}</span>`
        : "";

    banner.innerHTML = `
      <div class="answer-top">
        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">${statusTag}${droppedInfo}</div>
        <span class="mono-kicker">Grounding Score: ${Number(ans.grounding_score).toFixed(2)}</span>
      </div>
      <p class="answer-text">${highlightCitations(ans.text)}</p>
    `;

    // 2. Side-by-Side 2024 vs 2026 Conflict Diff
    const diffWrap = document.getElementById("conflictDiffContainer");
    const cp = res.conflict_side_by_side;
    if (cp) {
      const s24 = cp.superseded_2024_doc;
      const a26 = cp.active_2026_doc;
      diffWrap.innerHTML = `
        <div class="mono-kicker" style="margin-bottom:6px;">VERSION CONFLICT DETECTED &amp; RESOLVED PRE-SYNTHESIS (TOP-1 HIT RATE 75% &rarr; 100%):</div>
        <div class="conflict-diff-grid">
          <div class="diff-card stale">
            <span class="tag amber">SUPERSEDED ${escapeHtml(s24.effective_date.slice(0, 4))} · TIER-${s24.authority_tier} EXCLUDED</span>
            <div class="diff-title"><code>${escapeHtml(s24.doc_id)}</code> · ${escapeHtml(s24.title)}</div>
            <p class="diff-body">${escapeHtml(s24.content)}</p>
            <div class="src-meta" style="margin-top:6px;">Stale numbers blocked: ${escapeHtml((s24.numbers || []).join(", "))}</div>
          </div>
          <div class="diff-card active2026">
            <span class="tag ok">ACTIVE ${escapeHtml(a26.effective_date.slice(0, 4))} · TIER-${a26.authority_tier} AUTHORITATIVE</span>
            <div class="diff-title"><code>${escapeHtml(a26.doc_id)}</code> · ${escapeHtml(a26.title)}</div>
            <p class="diff-body">${escapeHtml(a26.content)}</p>
            <div class="src-meta" style="margin-top:6px;">Verified thresholds enforced: <strong>${escapeHtml((a26.numbers || []).join(", "))}</strong></div>
          </div>
        </div>
      `;
    } else {
      diffWrap.innerHTML = "";
    }

    // 3. Claim-by-Claim Verification Breakdown
    const claimsEl = document.getElementById("claimsList");
    claimsEl.innerHTML = (res.claims_breakdown || [])
      .map((c) => {
        const isAbs = c.status === "SAFE_ABSTENTION_ENFORCED";
        const quants = (c.quantitative_tokens || []).length
          ? `<div class="src-meta" style="margin-top:5px;">Quantitative tokens checked: <strong>${escapeHtml(c.quantitative_tokens.join(" · "))}</strong></div>`
          : "";
        return `<div class="claim-card ${isAbs ? "abstain-claim" : ""}">
          <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;">
            <span class="mono-kicker">CLAIM #${c.claim_index}</span>
            <span class="tag ${isAbs ? "amber" : "ok"}">${escapeHtml(c.status.replace(/_/g, " "))}</span>
          </div>
          <p style="font:400 0.98rem/1.5 var(--read);margin:6px 0;color:var(--ink);">${highlightCitations(c.claim_text)}</p>
          ${quants}
          <div class="src-meta" style="margin-top:5px;border-top:1px dashed var(--rule);padding-top:5px;">
            Provenance: ${c.matched_source_id ? `<code>${escapeHtml(c.matched_source_id)}</code> — ` : ""}${escapeHtml(c.matched_source_span)}
          </div>
        </div>`;
      })
      .join("");

    // 4. Retrieved Source Cards
    const srcEl = document.getElementById("sourcesList");
    srcEl.innerHTML = (res.source_cards || [])
      .map((s) => {
        const tagCls = s.badge_type === "SUPERSEDED_2024" ? "amber" : s.badge_type.startsWith("ACTIVE") ? "ok" : "cobalt";
        return `<div class="source-card ${escapeHtml(s.badge_type)}">
          <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;">
            <span class="tag ${tagCls}">${escapeHtml(s.badge_label)}</span>
            <span class="src-meta">RRF ${s.rrf_final_score} · Freshness &times;${s.freshness_multiplier}</span>
          </div>
          <div class="src-title"><code>${escapeHtml(s.doc_id)}</code> · ${escapeHtml(s.title)}</div>
          <p class="src-text">${escapeHtml(s.text)}</p>
          <div class="src-meta">Effective ${escapeHtml(s.effective_date)} · Region ${escapeHtml(s.region)} · Span chars [${s.char_offsets[0]}..${s.char_offsets[1]}]</div>
        </div>`;
      })
      .join("");

    // 5. 5-Stage Pipeline Trace
    document.getElementById("stageDagStrip").innerHTML = (res.stage_trace || [])
      .map(
        (st) => `<div class="dag-node">
          <strong>${escapeHtml(st.stage)}</strong>
          <span class="tag ${st.status.includes("BLOCKED") || st.status.includes("DROPPED") || st.status.includes("ABSTAIN") ? "amber" : "ok"}" style="margin-bottom:4px;">${escapeHtml(st.status)}</span>
          <div>${escapeHtml(st.detail)}</div>
        </div>`
      )
      .join("");

    // 6. Live Query Shadow Diff inside Tab B
    const sd = res.shadow_live_diff;
    if (sd) {
      document.getElementById("liveQueryShadowDiff").innerHTML = `
        <div class="diff-card stale">
          <span class="tag bad">PRODUCTION BASELINE · ${escapeHtml(sd.production_naive_rag.pipeline)}</span>
          <div class="diff-title" style="color:var(--bad);margin-top:8px;">Defect: ${escapeHtml(sd.production_naive_rag.defect_detected)}</div>
          <p style="font:400 0.96rem/1.5 var(--read);color:var(--body);">${escapeHtml(sd.production_naive_rag.answer)}</p>
          <div class="src-meta">Cited: ${escapeHtml((sd.production_naive_rag.cited_doc_ids || []).join(", ") || "None")}</div>
        </div>
        <div class="diff-card active2026">
          <span class="tag ok">ASYNC SHADOW CANDIDATE · ${escapeHtml(sd.candidate_verirag.pipeline)}</span>
          <div class="diff-title" style="color:var(--ok);margin-top:8px;">Verdict: ${escapeHtml(sd.candidate_verirag.verdict)} (+${sd.caller_added_latency_ms}ms caller impact)</div>
          <p style="font:400 0.96rem/1.5 var(--read);color:var(--ink);">${highlightCitations(sd.candidate_verirag.answer)}</p>
          <div class="src-meta">Cited: ${escapeHtml((sd.candidate_verirag.cited_doc_ids || []).join(", ") || "NONE (Safe Abstain)")}</div>
        </div>
      `;
    }
  }

  function renderShadowSimulationReport(rep) {
    if (!rep) return;
    const fchr = rep.fchr_bootstrap;
    const cit = rep.citation_precision_bootstrap;
    const cases = rep.replay_case_studies || [];

    document.getElementById("shadowBootstrapOutput").innerHTML = `
      <div class="stats" style="margin:8px 0 18px;border-top-width:1px;">
        <div><b>${rep.shadowed_queries_count} / ${rep.total_queries_received}</b><span>Queries mirrored (${(rep.actual_sample_rate * 100).toFixed(1)}% sample)</span></div>
        <div><b>${rep.mean_primary_caller_overhead_ms.toFixed(4)} ms</b><span>Primary caller added latency</span></div>
        <div><b>${(fchr.mean_delta * 100).toFixed(1)}%</b><span>Hallucination delta (p = ${fchr.p_value})</span></div>
        <div><b>+${(cit.mean_delta * 100).toFixed(1)}%</b><span>Citation precision gain (p = ${cit.p_value})</span></div>
      </div>
      <div class="mono-kicker" style="margin-bottom:8px;">COUNTERFACTUAL 5-STAGE REPLAY ROOT-CAUSE ISOLATION:</div>
      <div class="tablewrap" style="margin:0;"><table>
        <thead><tr><th>Trial ID</th><th>Original Defect</th><th>Isolated Stage</th><th>Node Swap</th></tr></thead>
        <tbody>
          ${cases
            .map(
              (c) => `<tr>
                <td><code>${escapeHtml(c.trial_id)}</code></td>
                <td>${escapeHtml(c.original_defect)}</td>
                <td><span class="tag cobalt">${escapeHtml(c.earliest_resolving_stage)}</span></td>
                <td><code>${escapeHtml(c.original_node)}</code> &rarr; <code>${escapeHtml(c.replacement_node)}</code></td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table></div>
    `;
  }

  async function executeVerify(useBedrock) {
    const btn = document.getElementById("verifySubmitBtn");
    const q = document.getElementById("verifyQueryInput").value.trim();
    const reg = document.getElementById("regionSelect").value;
    btn.disabled = true;
    btn.textContent = "Verifying via Bedrock...";
    try {
      const resp = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          domain_id: state.activeDomainId,
          query: q,
          region: reg,
          use_live_bedrock: Boolean(useBedrock),
        }),
      });
      renderVerificationResult(await resp.json());
    } finally {
      btn.disabled = false;
      btn.textContent = "Verify & Ground →";
    }
  }

  async function runShadowSim(rate) {
    const resp = await fetch("/api/shadow/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_rate: rate }),
    });
    renderShadowSimulationReport(await resp.json());
  }

  function setTab(which) {
    state.activeTab = which;
    document.getElementById("tabGrounding").classList.toggle("active", which === "grounding");
    document.getElementById("tabShadow").classList.toggle("active", which === "shadow");
    document.getElementById("paneGrounding").hidden = which !== "grounding";
    document.getElementById("paneShadow").hidden = which !== "shadow";
  }

  async function init() {
    const resp = await fetch("/api/bootstrap");
    const boot = await resp.json();
    state.domains = boot.domains || [];
    state.benchmarkSummary = boot.benchmark_summary || {};

    if (state.domains.length > 0) {
      document.getElementById("verifyQueryInput").value = state.domains[0].presets[0].query;
    }
    renderDomainsAndPresets();
    renderVerificationResult(boot.initial_verification);
    updateScrollChrome();

    // Immediately upgrade initial answer with live Amazon Bedrock Nova synthesis!
    executeVerify(true);
    // Pre-populate Shadow Router simulation report
    runShadowSim(1.0);

    document.getElementById("verifyQueryForm").addEventListener("submit", (e) => {
      e.preventDefault();
      state.activePresetId = "";
      renderDomainsAndPresets();
      executeVerify(true);
    });

    document.getElementById("tabGrounding").addEventListener("click", () => setTab("grounding"));
    document.getElementById("tabShadow").addEventListener("click", () => setTab("shadow"));
    document.getElementById("btnShadow5").addEventListener("click", () => runShadowSim(0.05));
    document.getElementById("btnShadow100").addEventListener("click", () => runShadowSim(1.0));
  }

  init();
})();
