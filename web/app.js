/* VeriRAG Editorial Redline Desk Controller — stateless in-memory UI */
(function () {
  "use strict";

  const state = {
    domains: [],
    activeDomainId: "rufus_catalog",
    activePresetId: "rufus-powerbank",
    region: "GLOBAL",
    shadowMode: 1.0,
    currentResult: null,
  };

  function esc(val) {
    return String(val == null ? "" : val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatRedlinedAnswer(text, abstained) {
    const safe = esc(text).replace(
      /\[(DOC-[A-Z0-9-]+)\]/g,
      '<span class="cite-tag">[$1]</span>'
    );
    if (abstained) {
      return `<span class="hl-rose">${safe}</span>`;
    }
    return `<span class="hl-verified">${safe}</span>`;
  }

  function renderFoldersAndPresets() {
    const folderWrap = document.getElementById("domainFolders");
    folderWrap.innerHTML = state.domains
      .map((d) => {
        const on = d.id === state.activeDomainId;
        return `<button type="button" class="folder-tab ${on ? "on" : ""}" data-did="${esc(d.id)}">
          <span class="folder-kicker">${esc(d.badge)}</span>
          <span class="folder-name">${esc(d.short_title)}</span>
        </button>`;
      })
      .join("");

    folderWrap.querySelectorAll(".folder-tab").forEach((b) => {
      b.addEventListener("click", () => {
        state.activeDomainId = b.getAttribute("data-did");
        const dom = state.domains.find((x) => x.id === state.activeDomainId);
        if (dom && dom.presets && dom.presets[0]) {
          const p0 = dom.presets[0];
          state.activePresetId = p0.id;
          state.region = p0.region || "GLOBAL";
          document.getElementById("regionSelect").value = state.region;
          document.getElementById("verifyQueryInput").value = p0.query;
          renderFoldersAndPresets();
          runVerification(true);
        }
      });
    });

    const curDom = state.domains.find((x) => x.id === state.activeDomainId);
    const pWrap = document.getElementById("presetButtons");
    if (curDom) {
      pWrap.innerHTML = (curDom.presets || [])
        .map((p) => {
          const on = p.id === state.activePresetId;
          return `<button type="button" class="case-chip ${on ? "on" : ""}" data-pid="${esc(p.id)}">${esc(p.label)}</button>`;
        })
        .join("");
      pWrap.querySelectorAll(".case-chip").forEach((b) => {
        b.addEventListener("click", () => {
          const pid = b.getAttribute("data-pid");
          const pObj = curDom.presets.find((x) => x.id === pid);
          if (!pObj) return;
          state.activePresetId = pObj.id;
          state.region = pObj.region || "GLOBAL";
          document.getElementById("regionSelect").value = state.region;
          document.getElementById("verifyQueryInput").value = pObj.query;
          renderFoldersAndPresets();
          runVerification(true);
        });
      });
    }
  }

  function renderVerification(res) {
    if (!res) return;
    state.currentResult = res;
    const ans = res.answer;
    const llm = ans.llm_telemetry || {};

    document.getElementById("liveEngineBadge").innerHTML =
      `${esc(llm.engine || "aws-bedrock (nova-lite-v1:0)")} · <strong>${res.wall_time_ms} ms</strong>`;

    // Column 1: Conflict Callout + Source Dossier Sheets
    const cp = res.conflict_side_by_side;
    const diffEl = document.getElementById("conflictDiffContainer");
    if (cp) {
      const s24 = cp.superseded_2024_doc;
      const a26 = cp.active_2026_doc;
      diffEl.innerHTML = `
        <div class="conflict-callout">
          <strong>⚡ Version Conflict Caught Pre-Synthesis:</strong>
          <code>${esc(s24.doc_id)}</code> excluded in favor of <code>${esc(a26.doc_id)}</code>.
          <div class="conflict-vs">
            <div class="vs-box stale">2024 Stale: ${esc((s24.numbers || []).slice(0, 3).join(", "))}</div>
            <div class="vs-box fresh">2026 Active: ${esc((a26.numbers || []).slice(0, 3).join(", "))}</div>
          </div>
        </div>`;
    } else {
      diffEl.innerHTML = "";
    }

    document.getElementById("sourcesList").innerHTML = (res.source_cards || [])
      .map((s) => {
        let stampHtml = `<span class="stamp stamp-neutral">DISTRACTOR · ${esc(s.effective_date.slice(0, 4))}</span>`;
        if (s.badge_type === "SUPERSEDED_2024") {
          stampHtml = `<span class="stamp stamp-superseded">SUPERSEDED · ${esc(s.effective_date.slice(0, 4))}</span>`;
        } else if (s.badge_type.startsWith("ACTIVE")) {
          stampHtml = `<span class="stamp stamp-active">ACTIVE · ${esc(s.effective_date.slice(0, 4))} TIER-${s.authority_tier}</span>`;
        }

        let passageHtml = esc(s.text);
        if (s.is_cited && s.verbatim_span && s.text.includes(s.verbatim_span)) {
          passageHtml = esc(s.text).replace(
            esc(s.verbatim_span),
            `<mark class="hl-verified">${esc(s.verbatim_span)}</mark>`
          );
        }

        return `<article class="dossier-card ${esc(s.badge_type)}">
          <div class="dossier-top">
            ${stampHtml}
            <span class="mono-note">RRF ${s.rrf_final_score} · &times;${s.freshness_multiplier}</span>
          </div>
          <h3 class="dossier-title"><code>${esc(s.doc_id)}</code> ${esc(s.title)}</h3>
          <p class="dossier-passage">${passageHtml}</p>
          <div class="dossier-foot">Effective ${esc(s.effective_date)} · ${esc(s.region)} · Offsets [${s.char_offsets[0]}..${s.char_offsets[1]}]</div>
        </article>`;
      })
      .join("");

    // Column 2: Redlined Proof Sheet & Claim Verifier
    const banner = document.getElementById("answerBanner");
    banner.className = "proof-sheet" + (ans.abstained ? " abstained" : "");
    const mainStamp = ans.abstained
      ? `<span class="stamp stamp-superseded">ABSTAINED · MISSING EVIDENCE</span>`
      : `<span class="stamp stamp-active">SPAN VERIFIED · GROUNDED</span>`;
    const droppedNote =
      ans.superseded_docs_dropped && ans.superseded_docs_dropped.length
        ? `<span class="mono-note" style="color:var(--crimson)">Redlined out: ${esc(ans.superseded_docs_dropped.join(", "))}</span>`
        : `<span class="mono-note">Grounding: ${Number(ans.grounding_score).toFixed(2)}</span>`;

    banner.innerHTML = `
      <div class="proof-topline">
        ${mainStamp}
        ${droppedNote}
      </div>
      <p class="proof-copy">${formatRedlinedAnswer(ans.text, ans.abstained)}</p>
    `;

    document.getElementById("claimsList").innerHTML = (res.claims_breakdown || [])
      .map((c) => {
        const isAbs = c.status === "SAFE_ABSTENTION_ENFORCED";
        const qToks = (c.quantitative_tokens || [])
          .map((t) => `<span class="q-tok">✓ ${esc(t)}</span>`)
          .join("");
        return `<div class="claim-row">
          <div class="claim-top">
            <span class="mono-note">CLAIM #${c.claim_index}</span>
            <span class="stamp ${isAbs ? "stamp-superseded" : "stamp-active"}" style="transform:none;font-size:9px;">${esc(c.status.replace(/_/g, " "))}</span>
          </div>
          <div style="font-size:13.5px;margin-bottom:4px;">${esc(c.claim_text).replace(/\[(DOC-[A-Z0-9-]+)\]/g, '<span class="cite-tag">[$1]</span>')}</div>
          ${qToks ? `<div class="quant-pills">${qToks}</div>` : ""}
          <div class="mono-note" style="font-size:10.5px;">Source span: ${c.matched_source_id ? `<code>${esc(c.matched_source_id)}</code> — ` : ""}${esc(c.matched_source_span)}</div>
        </div>`;
      })
      .join("");

    document.getElementById("stageDagStrip").innerHTML = (res.stage_trace || [])
      .map((st) => {
        const shortName = st.stage.replace(/^\d+\.\s*/, "").split("_").slice(0, 2).join(" ");
        return `<div class="pipe-step">
          <b>${esc(shortName)}</b>
          <span style="color:${st.status.includes("ABSTAIN") || st.status.includes("DROPPED") ? "var(--crimson)" : "var(--emerald)"}">${esc(st.status)}</span>
        </div>`;
      })
      .join("");

    // Column 3: Live Query Shadow Diff
    const sd = res.shadow_live_diff;
    if (sd) {
      document.getElementById("liveQueryShadowDiff").innerHTML = `
        <div class="shadow-box naive">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:12px;">Unguarded Naive Top-K RAG</strong>
            <span class="stamp stamp-superseded" style="font-size:8.5px;">CAUGHT DEFECT</span>
          </div>
          <div class="shadow-defect">${esc(sd.production_naive_rag.defect_detected)}</div>
          <div style="font-size:12.5px;color:#4c0519;">${esc(sd.production_naive_rag.answer)}</div>
        </div>
        <div class="shadow-box candidate">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:12px;">Candidate VeriRAG (Shadow)</strong>
            <span class="stamp stamp-active" style="font-size:8.5px;">+${sd.caller_added_latency_ms}ms</span>
          </div>
          <div style="font-size:12.5px;margin-top:5px;color:#064e3b;">${esc(sd.candidate_verirag.answer)}</div>
        </div>
      `;
    }
  }

  function renderShadowSuite(rep) {
    if (!rep) return;
    const fchr = rep.fchr_bootstrap;
    const cit = rep.citation_precision_bootstrap;
    const cases = (rep.replay_case_studies || []).slice(0, 3);

    document.getElementById("shadowBootstrapOutput").innerHTML = `
      <div class="boot-grid">
        <div class="boot-cell"><small>Shadow Sampled</small><b>${rep.shadowed_queries_count} / ${rep.total_queries_received}</b></div>
        <div class="boot-cell"><small>Caller Overhead</small><b style="color:var(--emerald)">${rep.mean_primary_caller_overhead_ms.toFixed(4)} ms</b></div>
        <div class="boot-cell"><small>Hallucination &Delta;</small><b style="color:var(--crimson)">${(fchr.mean_delta * 100).toFixed(1)}% (p=${fchr.p_value})</b></div>
        <div class="boot-cell"><small>Citation Gain</small><b style="color:var(--emerald)">+${(cit.mean_delta * 100).toFixed(1)}%</b></div>
      </div>
      <div class="mono-note" style="margin-bottom:6px;">5-STAGE REPLAY NODE-SWAP ISOLATION:</div>
      <div class="replay-list">
        ${cases
          .map(
            (c) => `<div class="replay-card">
              <strong>${esc(c.trial_id)}</strong> &rarr; <span style="color:var(--emerald)">[${esc(c.earliest_resolving_stage)}]</span><br>
              Swapped <code>${esc(c.original_node)}</code> &rarr; <code>${esc(c.replacement_node)}</code>
            </div>`
          )
          .join("")}
      </div>
    `;
  }

  async function runVerification(useBedrock) {
    const btn = document.getElementById("verifySubmitBtn");
    const q = document.getElementById("verifyQueryInput").value.trim();
    const reg = document.getElementById("regionSelect").value;
    btn.disabled = true;
    btn.textContent = "Redlining...";
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
      renderVerification(await resp.json());
    } finally {
      btn.disabled = false;
      btn.textContent = "Redline & Verify →";
    }
  }

  async function runShadow(rate) {
    state.shadowMode = rate;
    document.getElementById("btnShadow5").classList.toggle("on", rate < 0.5);
    document.getElementById("btnShadow100").classList.toggle("on", rate >= 0.5);
    const resp = await fetch("/api/shadow/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_rate: rate }),
    });
    renderShadowSuite(await resp.json());
  }

  async function init() {
    const resp = await fetch("/api/bootstrap");
    const boot = await resp.json();
    state.domains = boot.domains || [];
    if (state.domains.length > 0) {
      document.getElementById("verifyQueryInput").value = state.domains[0].presets[0].query;
    }
    renderFoldersAndPresets();
    renderVerification(boot.initial_verification);

    // Upgrade initial answer with live Amazon Bedrock Nova synthesis
    runVerification(true);
    runShadow(1.0);

    document.getElementById("verifyQueryForm").addEventListener("submit", (e) => {
      e.preventDefault();
      state.activePresetId = "";
      renderFoldersAndPresets();
      runVerification(true);
    });
    document.getElementById("btnShadow5").addEventListener("click", () => runShadow(0.05));
    document.getElementById("btnShadow100").addEventListener("click", () => runShadow(1.0));
  }

  init();
})();
