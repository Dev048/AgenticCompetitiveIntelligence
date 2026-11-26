/**
 * VANTAGE // Ask the Evidence
 * Workspace-scoped question answering over indexed evidence, showing the agentic steps:
 * escalation decision, query analysis, sub-queries, verification verdicts, contradictions,
 * and the cited chunks with their provenance pointers. No free-text answer is generated:
 * the output is grounded evidence plus the audit trace.
 */

import { api } from '../api.js';
import { state } from '../state.js';

const EXAMPLES = [
  'How did Northwind Metrics change its pricing?',
  'Does the Northwind Free plan really include 3 seats?',
  'What enterprise moves is Quarry Labs making?',
  'What did Helios Insight ship in release 8.0?',
];

export async function renderAsk(container) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  container.innerHTML = `
    <div style="margin-bottom:18px;">
      <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Ask the Evidence</h2>
      <p style="color:var(--text-secondary); font-size:13px; margin-top:4px; max-width:760px; line-height:1.55;">
        Questions run against this workspace's indexed sources. <strong>Auto</strong> uses cheap deterministic
        hybrid retrieval and escalates to the bounded agentic graph only when it detects an evidence gap.
        <strong>Agentic</strong> always runs it: query analysis, sub-queries, sufficiency and contradiction checks.
      </p>
    </div>

    <div class="card">
      <form class="ask-form" id="ask-form">
        <div class="ask-input-row">
          <input class="form-control" id="ask-input" maxlength="500" minlength="3" required
                 placeholder="e.g. Does the Northwind Free plan really include 3 seats?" aria-label="Question" />
          <button class="btn btn-primary" type="submit" id="ask-submit">Ask</button>
        </div>
        <div class="ask-mode" role="radiogroup" aria-label="Retrieval mode">
          <label><input type="radio" name="ask-mode" value="auto" checked /> Auto (escalate only if needed)</label>
          <label><input type="radio" name="ask-mode" value="agentic" /> Agentic (always show the reasoning steps)</label>
        </div>
        <div class="ask-examples">
          ${EXAMPLES.map((q) => `<button type="button" class="ask-chip" data-q="${state.escapeHtml(q)}">${state.escapeHtml(q)}</button>`).join('')}
        </div>
      </form>
    </div>

    <div id="ask-output"></div>
  `;

  const form = container.querySelector('#ask-form');
  const input = container.querySelector('#ask-input');
  const output = container.querySelector('#ask-output');
  const submit = container.querySelector('#ask-submit');

  container.querySelectorAll('.ask-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.q;
      form.requestSubmit();
    });
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (query.length < 3) return;
    const mode = form.querySelector('input[name="ask-mode"]:checked')?.value || 'auto';

    submit.disabled = true;
    submit.textContent = 'Searching…';
    output.innerHTML = `<div class="card skeleton" style="height:220px; margin-top:20px;"></div>`;
    try {
      const result = await api.queryEvidence(wsId, { query_text: query, mode, top_k: 5 });
      renderResult(output, result);
    } catch (err) {
      output.innerHTML = `<div class="notice ${err.status === 429 ? '' : 'notice-danger'}" style="margin-top:20px;">
        ${state.escapeHtml(friendlyError(err))}
      </div>`;
    } finally {
      submit.disabled = false;
      submit.textContent = 'Ask';
    }
  });
}

function friendlyError(err) {
  if (err.status === 429) return err.message || 'Too many live queries right now. Please wait a moment and retry.';
  if (err.status === 403) return err.message || 'This action is disabled in the read-only demo.';
  if (err.status === 422) return 'Please enter a question between 3 and 500 characters.';
  return `The query failed: ${err.message || 'unknown error'}`;
}

function renderResult(output, result) {
  const t = result.trace || {};
  const esc = (v) => state.escapeHtml(v == null ? '' : String(v));

  const steps = [];
  steps.push(`
    <div class="trace-step">
      <div class="trace-step-title">1 · Escalation policy
        <span class="badge ${t.escalated ? 'badge-purple' : 'badge-cyan'}">${t.escalated ? 'escalated to agentic' : 'deterministic only'}</span>
      </div>
      ${t.escalation_reasons?.length
        ? `<ul>${t.escalation_reasons.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>`
        : '<div>Initial hybrid retrieval looked sufficient, so no LLM calls were needed.</div>'}
    </div>`);

  if (t.query_decision) {
    steps.push(`
      <div class="trace-step">
        <div class="trace-step-title">2 · Query analysis <span class="badge badge-neutral">${esc(t.query_decision.replace(/_/g, ' '))}</span></div>
        ${t.query_reasoning ? `<div style="margin-top:4px;">${esc(t.query_reasoning)}</div>` : ''}
        ${t.sub_queries?.length ? `<ul>${t.sub_queries.map((q) => `<li><strong>${esc(q.sub_query)}</strong>${q.rationale ? ` – ${esc(q.rationale)}` : ''}</li>`).join('')}</ul>` : ''}
      </div>`);
  }

  (t.verifications || []).forEach((v, idx) => {
    steps.push(`
      <div class="trace-step">
        <div class="trace-step-title">${3 + idx} · Verification, hop ${idx + 1}
          <span class="badge ${v.is_sufficient ? 'badge-success' : 'badge-warning'}">${esc(v.decision.replace(/_/g, ' '))}</span>
        </div>
        <div style="margin-top:4px;">${esc(v.reasoning)}</div>
        ${v.missing_aspects?.length ? `<div style="margin-top:4px;">Missing: ${v.missing_aspects.map(esc).join('; ')}</div>` : ''}
        ${v.contradictions?.length ? `<ul>${v.contradictions.map((c) => `<li><strong>Contradiction:</strong> ${esc(c.claim_description)}${c.rationale ? ` – ${esc(c.rationale)}` : ''}</li>`).join('')}</ul>` : ''}
      </div>`);
  });

  if (t.termination_reason) {
    steps.push(`<div class="trace-step"><div class="trace-step-title">Stopped</div><div>${esc(t.termination_reason)}</div></div>`);
  }

  output.innerHTML = `
    ${t.degraded ? `<div class="notice" style="margin-top:20px;">${esc(t.degraded_reason)}</div>` : ''}
    <div class="ask-results">
      <div class="card">
        <div class="card-header"><h3 class="card-title">How the evidence was gathered</h3></div>
        <div class="trace-stats" style="margin-bottom:12px;">
          <div class="trace-stat"><div class="trace-stat-value">${esc(t.llm_calls ?? 0)}</div><div class="trace-stat-label">LLM calls</div></div>
          <div class="trace-stat"><div class="trace-stat-value">${esc(t.retrieval_hops ?? 1)}</div><div class="trace-stat-label">Hops</div></div>
          <div class="trace-stat"><div class="trace-stat-value">${esc(Math.round(t.latency_ms ?? 0))}</div><div class="trace-stat-label">ms</div></div>
          <div class="trace-stat"><div class="trace-stat-value" style="font-size:12px;">${esc((t.final_source || '').replace(/_/g, ' '))}</div><div class="trace-stat-label">Answered by</div></div>
        </div>
        <div class="trace-steps">${steps.join('')}</div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Cited evidence</h3>
          <span class="badge badge-neutral">${esc(result.evidence.length)} chunks</span>
        </div>
        ${result.evidence.length === 0
          ? '<div class="empty-description">No matching evidence in this workspace.</div>'
          : result.evidence.map((e, i) => `
            <div class="evidence-card">
              <div style="display:flex; justify-content:space-between; gap:8px; margin-bottom:6px;">
                <span class="badge badge-cyan">#${i + 1} · score ${Number(e.score).toFixed(3)}</span>
                ${e.canonical_url ? `<span class="prov-meta" style="margin:0;">${esc(e.canonical_url)}</span>` : ''}
              </div>
              <div class="evidence-text">${esc(e.text)}</div>
              <div class="prov-meta">
                <span>chunk ${esc(e.chunk_id.slice(0, 12))}…</span>
                <span>version ${esc(e.version_id.slice(0, 12))}…</span>
                ${e.source_id ? `<span>source ${esc(e.source_id)}</span>` : ''}
              </div>
            </div>`).join('')}
      </div>
    </div>
  `;
}
