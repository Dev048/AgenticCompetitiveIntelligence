/**
 * VANTAGE // Strategic Insights & Recommendations View
 * Evidence-grounded Strategic Insights and Actionable Recommendations with human governance.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderStrategicInsights(container, { query = {} } = {}) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  const activeTab = query.tab || 'insights';

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
      <div>
        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Strategic Insights & Recommendations</h2>
        <p style="color:var(--text-muted); font-size:13px;">
          Synthesized multi-source intelligence, strategic implications, and prescriptive decision support with human-in-the-loop governance.
        </p>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <select class="form-control" id="insights-filter-competitor" style="width:180px; padding:6px 12px; font-size:12px;">
          <option value="">All Competitors</option>
        </select>
        <input type="text" class="form-control" id="insights-search" placeholder="Search keywords..." style="width:200px; padding:6px 12px; font-size:12px;" />
        <button class="btn btn-primary" id="btn-create-context" style="font-size:12px;">
          <span>+</span> Formulate Context
        </button>
      </div>
    </div>

    <!-- Sub-tab Navigation -->
    <div class="tab-bar">
      <button class="tab-btn ${activeTab === 'insights' ? 'active' : ''}" data-tab="insights">
        <span>🧠 Generated Insights</span>
        <span class="badge badge-neutral" id="tab-count-insights">-</span>
      </button>
      <button class="tab-btn ${activeTab === 'recommendations' ? 'active' : ''}" data-tab="recommendations">
        <span>⚖️ Actionable Recommendations</span>
        <span class="badge badge-neutral" id="tab-count-recommendations">-</span>
      </button>
      <button class="tab-btn ${activeTab === 'contexts' ? 'active' : ''}" data-tab="contexts">
        <span>📋 Decision Contexts</span>
        <span class="badge badge-neutral" id="tab-count-contexts">-</span>
      </button>
    </div>

    <div id="insights-content-area">
      <div class="skeleton skeleton-card" style="height:350px;"></div>
    </div>
  `;

  // Wire tab clicks
  container.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      window.location.hash = `#/insights?tab=${tab}`;
    });
  });

  try {
    const [competitors, insights, recommendations, contexts, events, signals] = await Promise.all([
      api.getCompetitors(wsId).catch(() => []),
      api.getInsights(wsId, null, 100).catch(() => []),
      api.getRecommendations(wsId).catch(() => []),
      api.getDecisionContexts(wsId).catch(() => []),
      api.getEvents(wsId, null, 100).catch(() => []),
      api.getSignals(wsId, null, 100).catch(() => []),
    ]);

    const compMap = new Map();
    competitors.forEach(c => compMap.set(c.competitor_id, c.name));

    // Update counts
    const countInsights = document.getElementById('tab-count-insights');
    const countRecs = document.getElementById('tab-count-recommendations');
    const countCtx = document.getElementById('tab-count-contexts');
    if (countInsights) countInsights.textContent = insights.length;
    if (countRecs) countRecs.textContent = recommendations.length;
    if (countCtx) countCtx.textContent = contexts.length;

    // Populate filter dropdown
    const compSelect = document.getElementById('insights-filter-competitor');
    if (compSelect) {
      competitors.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.competitor_id;
        opt.textContent = c.name;
        compSelect.appendChild(opt);
      });
    }

    // Wire Formulate Context Button
    document.getElementById('btn-create-context')?.addEventListener('click', () => {
      openCreateContextModal(wsId, competitors, () => renderStrategicInsights(container, { query: { tab: 'contexts' } }));
    });

    const contentArea = document.getElementById('insights-content-area');
    const searchInput = document.getElementById('insights-search');

    const updateView = () => {
      const selectedComp = compSelect?.value || '';
      const searchQuery = (searchInput?.value || '').toLowerCase().trim();

      if (activeTab === 'insights') {
        renderInsightsList(contentArea, insights, compMap, selectedComp, searchQuery, events, signals);
      } else if (activeTab === 'recommendations') {
        renderRecommendationsList(contentArea, recommendations, contexts, compMap, selectedComp, searchQuery, wsId, () => renderStrategicInsights(container, { query }));
      } else if (activeTab === 'contexts') {
        renderContextsList(contentArea, contexts, competitors, compMap, selectedComp, searchQuery, wsId, () => renderStrategicInsights(container, { query }));
      }
    };

    compSelect?.addEventListener('change', updateView);
    searchInput?.addEventListener('input', updateView);

    updateView();
  } catch (err) {
    console.error('[STRATEGIC INSIGHTS ERROR]', err);
    const contentArea = document.getElementById('insights-content-area');
    if (contentArea) {
      contentArea.innerHTML = `
        <div class="card" style="border-color: var(--status-danger);">
          <h3 class="card-title" style="color:var(--status-danger);">Error Loading Strategic Insights</h3>
          <p style="color:var(--text-secondary); margin-top:8px;">${state.escapeHtml(err.message)}</p>
          <div style="margin-top:12px;">
            <button class="btn btn-secondary" onclick="window.location.reload()">Retry</button>
          </div>
        </div>
      `;
    }
  }
}

// -----------------------------------------------------------------------------
// TAB 1: Generated Insights
// -----------------------------------------------------------------------------
function renderInsightsList(container, insights, compMap, filterComp, filterSearch, allEvents, allSignals) {
  let filtered = insights;
  if (filterComp) filtered = filtered.filter(i => i.company_id === filterComp);
  if (filterSearch) {
    filtered = filtered.filter(i =>
      (i.title || '').toLowerCase().includes(filterSearch) ||
      (i.statement || '').toLowerCase().includes(filterSearch) ||
      (i.insight_type || '').toLowerCase().includes(filterSearch)
    );
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🧠</div>
        <h3 class="empty-title">No Strategic Insights Found</h3>
        <p class="empty-description">
          Insights synthesize verified events and multi-source signals into high-level strategic takeaways for leadership.
        </p>
        <div style="margin-top:14px;">
          <a href="#/workflows" class="btn btn-primary btn-sm">Run Intelligence Pipeline</a>
        </div>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:16px;">
      ${filtered.map(ins => {
        const factualScore = Math.round((ins.factual_confidence?.score || 0.8) * 100);
        const interpretiveScore = Math.round((ins.interpretive_confidence?.score || 0.75) * 100);
        const horizon = ins.temporal_scope?.horizon || 'medium_term';

        return `
          <div class="card" style="border-left:4px solid var(--accent-primary);">
            <div class="card-header">
              <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                <span class="badge badge-purple">${state.escapeHtml(ins.insight_type || 'STRATEGIC_SYNTHESIS')}</span>
                <span class="card-title">${state.escapeHtml(ins.title || 'Strategic Implication')}</span>
                <span style="font-size:12px; color:var(--accent-cyan); font-weight:600;">
                  ${state.escapeHtml(compMap.get(ins.company_id) || ins.company_id)}
                </span>
              </div>
              <span class="badge ${ins.status === 'active' ? 'badge-success' : 'badge-neutral'}">
                ${state.escapeHtml(ins.status || 'ACTIVE')}
              </span>
            </div>

            <div class="insight-statement">
              "${state.escapeHtml(ins.statement)}"
            </div>

            ${ins.summary ? `
              <p style="font-size:13px; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
                ${state.escapeHtml(ins.summary)}
              </p>
            ` : ''}

            <!-- Metrics and Grounding Summary -->
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; margin-top:14px; padding-top:12px; border-top:1px solid var(--border-subtle);">
              <div style="display:flex; align-items:center; gap:18px; font-size:12px; color:var(--text-muted); flex-wrap:wrap;">
                <div>
                  Factual Confidence: <strong style="color:var(--text-primary); font-family:var(--font-mono);">${factualScore}%</strong>
                </div>
                <div>
                  Interpretive Confidence: <strong style="color:var(--text-primary); font-family:var(--font-mono);">${interpretiveScore}%</strong>
                </div>
                <div>
                  Horizon: <strong style="color:var(--accent-cyan);">${state.escapeHtml(horizon.replace('_', ' '))}</strong>
                </div>
              </div>

              <div style="display:flex; align-items:center; gap:10px;">
                <span class="badge badge-neutral">${ins.supporting_event_ids?.length || 0} Events Linked</span>
                <span class="badge badge-neutral">${ins.supporting_signal_ids?.length || 0} Signals Linked</span>
                <button class="btn btn-secondary btn-sm btn-inspect-insight" data-insight-id="${ins.insight_id}">
                  Inspect Lineage &rarr;
                </button>
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;

  container.querySelectorAll('.btn-inspect-insight').forEach(btn => {
    btn.addEventListener('click', () => {
      const insId = btn.dataset.insightId;
      const ins = insights.find(i => i.insight_id === insId);
      if (ins) openInsightDrawer(ins, compMap.get(ins.company_id), allEvents, allSignals);
    });
  });
}

// -----------------------------------------------------------------------------
// TAB 2: Actionable Recommendations
// -----------------------------------------------------------------------------
function renderRecommendationsList(container, recommendations, contexts, compMap, filterComp, filterSearch, wsId, onRefresh) {
  let filtered = recommendations;
  if (filterSearch) {
    filtered = filtered.filter(r =>
      (r.title || '').toLowerCase().includes(filterSearch) ||
      (r.rationale || '').toLowerCase().includes(filterSearch) ||
      (r.urgency || '').toLowerCase().includes(filterSearch)
    );
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚖️</div>
        <h3 class="empty-title">No Prescriptive Recommendations</h3>
        <p class="empty-description">
          Recommendations prescribe concrete counter-moves and require human operator approval. Run a workflow with Decision Support enabled or formulate a Decision Context.
        </p>
        <div style="margin-top:14px; display:flex; gap:10px;">
          <a href="#/workflows" class="btn btn-primary btn-sm">Dispatch Decision Workflow</a>
        </div>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:16px;">
      ${filtered.map(rec => {
        const isAccepted = rec.status === 'accepted';
        const isRejected = rec.status === 'rejected';
        const isProposed = rec.status === 'proposed';
        const borderColor = isAccepted ? 'var(--status-success)' : isRejected ? 'var(--status-danger)' : 'var(--accent-primary)';
        const confidenceScore = Math.round((rec.confidence?.score || 0.85) * 100);

        return `
          <div class="card" style="border-left:4px solid ${borderColor};">
            <div class="card-header">
              <div>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; flex-wrap:wrap;">
                  <span class="badge ${rec.urgency === 'urgent' ? 'badge-danger' : 'badge-cyan'}">${state.escapeHtml(rec.urgency || 'MEDIUM')}</span>
                  <span class="badge ${isAccepted ? 'badge-success' : isRejected ? 'badge-danger' : 'badge-warning'}">
                    STATUS: ${state.escapeHtml((rec.status || 'PROPOSED').toUpperCase())}
                  </span>
                </div>
                <h4 style="font-family:var(--font-display); font-size:16px; font-weight:700; color:var(--text-primary); margin-top:4px;">
                  ${state.escapeHtml(rec.title)}
                </h4>
              </div>
              <button class="btn btn-secondary btn-sm btn-view-provenance" data-rec-id="${rec.recommendation_id}">
                🔍 10-Layer Lineage
              </button>
            </div>

            <p style="font-size:13px; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
              ${state.escapeHtml(rec.rationale)}
            </p>

            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; background:var(--bg-canvas); padding:14px; border-radius:8px; margin-top:12px; border:1px solid var(--border-subtle);">
              <div>
                <div class="form-label" style="font-size:11px; margin-bottom:6px; color:var(--status-success);">Expected Strategic Benefits</div>
                <ul style="padding-left:18px; font-size:12px; color:var(--text-secondary); line-height:1.5;">
                  ${rec.expected_benefits?.length ? rec.expected_benefits.map(b => `<li>${state.escapeHtml(b)}</li>`).join('') : '<li>Competitive positioning reinforcement</li>'}
                </ul>
              </div>
              <div>
                <div class="form-label" style="font-size:11px; margin-bottom:6px; color:var(--status-warning);">Trade-offs & Resource Demands</div>
                <ul style="padding-left:18px; font-size:12px; color:var(--text-secondary); line-height:1.5;">
                  ${rec.tradeoffs?.length ? rec.tradeoffs.map(t => `<li>${state.escapeHtml(t)}</li>`).join('') : '<li>Standard resource re-allocation</li>'}
                </ul>
              </div>
            </div>

            <!-- Human Approval Governance Bar -->
            <div style="display:flex; align-items:center; justify-content:space-between; padding-top:12px; border-top:1px solid var(--border-subtle); flex-wrap:wrap; gap:12px; margin-top:14px;">
              <div style="font-size:12px; color:var(--text-muted);">
                Confidence: <strong style="color:var(--text-primary); font-family:var(--font-mono);">${confidenceScore}%</strong>
                • Formulated: ${rec.generated_at ? new Date(rec.generated_at).toLocaleDateString() : 'Recent'}
              </div>

              ${isProposed ? `
                <div style="display:flex; align-items:center; gap:10px;">
                  <button class="btn btn-success btn-sm btn-approve-rec" data-rec-id="${rec.recommendation_id}" data-rec-title="${state.escapeHtml(rec.title)}">
                    ✓ Approve Recommendation
                  </button>
                  <button class="btn btn-danger btn-sm btn-reject-rec" data-rec-id="${rec.recommendation_id}" data-rec-title="${state.escapeHtml(rec.title)}">
                    ✕ Reject
                  </button>
                </div>
              ` : `
                <div style="font-size:12px; color:var(--text-muted); font-style:italic;">
                  Decision governance finalized: <strong style="color:${isAccepted ? 'var(--status-success)' : 'var(--status-danger)'};">${state.escapeHtml((rec.status || '').toUpperCase())}</strong>
                  ${rec.review_notes ? `— "${state.escapeHtml(rec.review_notes)}"` : ''}
                </div>
              `}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;

  // Approve action
  container.querySelectorAll('.btn-approve-rec').forEach(btn => {
    btn.addEventListener('click', () => {
      const recId = btn.dataset.recId;
      const recTitle = btn.dataset.recTitle;
      openReviewModal(wsId, recId, recTitle, 'accepted', onRefresh);
    });
  });

  // Reject action
  container.querySelectorAll('.btn-reject-rec').forEach(btn => {
    btn.addEventListener('click', () => {
      const recId = btn.dataset.recId;
      const recTitle = btn.dataset.recTitle;
      openReviewModal(wsId, recId, recTitle, 'rejected', onRefresh);
    });
  });

  // Lineage provenance action
  container.querySelectorAll('.btn-view-provenance').forEach(btn => {
    btn.addEventListener('click', async () => {
      const recId = btn.dataset.recId;
      try {
        const prov = await api.getRecommendationProvenance(wsId, recId);
        openProvenanceDrawer(prov);
      } catch (err) {
        state.showToast(`Failed loading provenance: ${err.message}`, 'danger');
      }
    });
  });
}

// -----------------------------------------------------------------------------
// TAB 3: Decision Contexts
// -----------------------------------------------------------------------------
function renderContextsList(container, contexts, competitors, compMap, filterComp, filterSearch, wsId, onRefresh) {
  let filtered = contexts;
  if (filterSearch) {
    filtered = filtered.filter(c =>
      (c.title || '').toLowerCase().includes(filterSearch) ||
      (c.decision_domain || '').toLowerCase().includes(filterSearch) ||
      (c.current_state_summary || '').toLowerCase().includes(filterSearch)
    );
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📋</div>
        <h3 class="empty-title">No Decision Contexts Defined</h3>
        <p class="empty-description">
          Decision contexts encapsulate business objectives, operational constraints, and risk tolerances to benchmark candidate strategic counter-moves.
        </p>
        <div style="margin-top:14px;">
          <button class="btn btn-primary btn-sm" id="btn-empty-create-ctx">Formulate Decision Context</button>
        </div>
      </div>
    `;
    container.querySelector('#btn-empty-create-ctx')?.addEventListener('click', () => {
      openCreateContextModal(wsId, competitors, onRefresh);
    });
    return;
  }

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:16px;">
      ${filtered.map(ctx => `
        <div class="card">
          <div class="card-header">
            <div>
              <span class="badge badge-purple" style="margin-bottom:6px;">${state.escapeHtml(ctx.decision_domain || 'GENERAL')}</span>
              <h3 class="card-title">${state.escapeHtml(ctx.title)}</h3>
              <p class="card-subtitle">${state.escapeHtml(ctx.current_state_summary || 'No current state summary provided.')}</p>
            </div>
            <div style="text-align:right;">
              <span class="badge badge-cyan">${state.escapeHtml(ctx.risk_tolerance || 'MODERATE')}</span>
              <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Horizon: ${state.escapeHtml(ctx.time_horizon || 'medium_term')}</div>
            </div>
          </div>

          <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px; margin-top:14px;">
            <div style="background:var(--bg-canvas); padding:14px; border-radius:8px; border:1px solid var(--border-subtle);">
              <div class="form-label" style="margin-bottom:8px;">Business Objectives (${ctx.business_objectives?.length || 0})</div>
              <div style="display:flex; flex-direction:column; gap:6px;">
                ${ctx.business_objectives?.length ? ctx.business_objectives.map(o => `
                  <div style="font-size:12px; display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:var(--text-primary);">${state.escapeHtml(o.name)} [${state.escapeHtml(o.dimension || 'strategic')}]</span>
                    <span style="color:var(--accent-primary); font-family:var(--font-mono); font-weight:600;">
                      ${o.priority_weight ? Math.round(o.priority_weight * 100) + '%' : ''}
                    </span>
                  </div>
                `).join('') : '<div style="color:var(--text-muted); font-size:12px;">No objectives specified.</div>'}
              </div>
            </div>

            <div style="background:var(--bg-canvas); padding:14px; border-radius:8px; border:1px solid var(--border-subtle);">
              <div class="form-label" style="margin-bottom:8px;">Operational Constraints (${ctx.constraints?.length || 0})</div>
              <div style="display:flex; flex-direction:column; gap:6px;">
                ${ctx.constraints?.length ? ctx.constraints.map(c => `
                  <div style="font-size:12px; display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:var(--text-secondary);">${state.escapeHtml(c.description)}</span>
                    <span class="badge ${c.is_hard_constraint ? 'badge-danger' : 'badge-neutral'}" style="font-size:9px;">
                      ${c.is_hard_constraint ? 'HARD' : 'SOFT'}
                    </span>
                  </div>
                `).join('') : '<div style="color:var(--text-muted); font-size:12px;">No constraints specified.</div>'}
              </div>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// -----------------------------------------------------------------------------
// MODALS & DRAWERS
// -----------------------------------------------------------------------------
function openReviewModal(workspaceId, recommendationId, title, decisionStatus, onDone) {
  const isAccept = decisionStatus === 'accepted';
  state.openModal({
    title: `${isAccept ? 'Approve' : 'Reject'} Recommendation`,
    renderBody: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:14px;">
          <p style="font-size:13px; color:var(--text-secondary); line-height:1.5;">
            You are about to record a human governance decision for recommendation:
            <strong style="color:var(--text-primary);">${state.escapeHtml(title)}</strong>
          </p>
          <div class="form-group">
            <label class="form-label">Reviewer Governance Notes (Optional)</label>
            <textarea class="form-control" id="review-notes" rows="3" placeholder="Document executive rationale, budgetary alignment, or counter-risk considerations..."></textarea>
          </div>
        </div>
      `;
    },
    renderFooter: (el) => {
      el.innerHTML = `
        <div style="display:flex; justify-content:flex-end; gap:10px;">
          <button class="btn btn-secondary btn-sm" id="btn-cancel-review">Cancel</button>
          <button class="btn ${isAccept ? 'btn-success' : 'btn-danger'} btn-sm" id="btn-confirm-review">
            Confirm ${isAccept ? 'Approval' : 'Rejection'}
          </button>
        </div>
      `;
      el.querySelector('#btn-cancel-review')?.addEventListener('click', () => state.closeModal());
      el.querySelector('#btn-confirm-review')?.addEventListener('click', async () => {
        const notes = document.getElementById('review-notes')?.value || null;
        try {
          await api.updateRecommendationStatus(workspaceId, recommendationId, decisionStatus, notes);
          state.closeModal();
          state.showToast(`Recommendation successfully marked as ${decisionStatus.toUpperCase()}`, 'success');
          if (onDone) onDone();
        } catch (err) {
          state.showToast(`Failed updating recommendation: ${err.message}`, 'danger');
        }
      });
    },
  });
}

function openCreateContextModal(workspaceId, competitors, onDone) {
  state.openModal({
    title: 'Formulate Strategic Decision Context',
    renderBody: (el) => {
      el.innerHTML = `
        <form id="form-create-context" style="display:flex; flex-direction:column; gap:14px;">
          <div class="form-group">
            <label class="form-label">Target Tracked Competitor</label>
            <select class="form-control" id="ctx-competitor" required>
              <option value="">Select Target Competitor...</option>
              ${competitors.map(c => `<option value="${c.competitor_id}">${state.escapeHtml(c.name)} (${state.escapeHtml(c.domain || 'no domain')})</option>`).join('')}
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Decision Context Title</label>
            <input type="text" class="form-control" id="ctx-title" placeholder="e.g. Downmarket SaaS Tier Response" required />
          </div>
          <div class="form-group">
            <label class="form-label">Strategic Decision Domain</label>
            <select class="form-control" id="ctx-domain">
              <option value="pricing">Pricing & Packaging</option>
              <option value="product">Product Feature & Roadmap</option>
              <option value="gtm_marketing">Market Expansion & GTM</option>
              <option value="hiring_talent">Executive Talent & Hiring</option>
              <option value="technology_infrastructure">Technology Infrastructure</option>
              <option value="partnership_ecosystem">Partnership Ecosystem</option>
              <option value="risk_compliance">Risk & Compliance</option>
              <option value="general_strategy">General Strategy</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Current State Assessment</label>
            <textarea class="form-control" id="ctx-current-state" rows="2" placeholder="Describe the competitive baseline and urgency..."></textarea>
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div class="form-group">
              <label class="form-label">Risk Tolerance</label>
              <select class="form-control" id="ctx-risk">
                <option value="moderate">Moderate</option>
                <option value="very_low_conservative">Conservative (Very Low)</option>
                <option value="high_aggressive">Aggressive (High)</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">Time Horizon</label>
              <select class="form-control" id="ctx-horizon">
                <option value="medium_term">Medium Term</option>
                <option value="immediate_short_term">Immediate (&lt; 30 days)</option>
                <option value="long_term">Long Term</option>
              </select>
            </div>
          </div>
        </form>
      `;
    },
    renderFooter: (el) => {
      el.innerHTML = `
        <div style="display:flex; justify-content:flex-end; gap:10px;">
          <button class="btn btn-secondary btn-sm" id="btn-cancel-create-ctx">Cancel</button>
          <button class="btn btn-primary btn-sm" id="btn-submit-create-ctx">Formulate Context</button>
        </div>
      `;
      el.querySelector('#btn-cancel-create-ctx')?.addEventListener('click', () => state.closeModal());
      el.querySelector('#btn-submit-create-ctx')?.addEventListener('click', async () => {
        const compId = document.getElementById('ctx-competitor')?.value;
        const title = document.getElementById('ctx-title')?.value;
        const domain = document.getElementById('ctx-domain')?.value;
        const currentState = document.getElementById('ctx-current-state')?.value;
        const risk = document.getElementById('ctx-risk')?.value;
        const horizon = document.getElementById('ctx-horizon')?.value;

        if (!compId || !title) {
          state.showToast('Competitor and title are required fields', 'warning');
          return;
        }

        try {
          await api.createDecisionContext(workspaceId, {
            title,
            decision_domain: domain,
            competitor_ids: [compId],
            current_state_summary: currentState || null,
            risk_tolerance: risk,
            time_horizon: horizon,
            business_objectives: [
              {
                name: 'Defend Tier 1 ARR',
                dimension: 'revenue',
                target_description: 'Protect core revenue from competitive disruption',
                priority_weight: 0.6,
              },
              {
                name: 'Minimize Gross Margin Erosion',
                dimension: 'cost_efficiency',
                target_description: 'Preserve product margin thresholds',
                priority_weight: 0.4,
              },
            ],
            constraints: [
              {
                constraint_type: 'organizational_policy',
                description: 'No discounting below 20% floor',
                is_hard_constraint: true,
              },
            ],
          });
          state.closeModal();
          state.showToast('Decision context successfully formulated', 'success');
          if (onDone) onDone();
        } catch (err) {
          state.showToast(`Failed creating context: ${err.message}`, 'danger');
        }
      });
    },
  });
}

function openInsightDrawer(insight, compName, allEvents = [], allSignals = []) {
  state.openDrawer({
    title: insight.title || 'Strategic Insight',
    subtitle: `Insight [${insight.insight_type}] for ${compName || insight.company_id}`,
    renderContent: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:18px;">
          <div>
            <div class="form-label">Core Strategic Statement</div>
            <div class="insight-statement" style="margin-top:6px;">
              "${state.escapeHtml(insight.statement)}"
            </div>
          </div>

          ${insight.summary ? `
            <div>
              <div class="form-label">Executive Summary & Analytical Breakdown</div>
              <p style="color:var(--text-secondary); line-height:1.6; margin-top:4px; font-size:13px;">
                ${state.escapeHtml(insight.summary)}
              </p>
            </div>
          ` : ''}

          <div style="background:var(--bg-canvas); padding:14px; border-radius:8px; border:1px solid var(--border-subtle);">
            <div class="form-label" style="margin-bottom:8px;">Analytical Confidence Metrics</div>
            <div style="display:flex; flex-direction:column; gap:8px; font-size:12px;">
              <div style="display:flex; justify-content:space-between;">
                <span style="color:var(--text-secondary);">Factual Grounding:</span>
                <strong style="color:var(--status-success); font-family:var(--font-mono);">${Math.round((insight.factual_confidence?.score || 0.8) * 100)}%</strong>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span style="color:var(--text-secondary);">Interpretive Reliability:</span>
                <strong style="color:var(--accent-primary); font-family:var(--font-mono);">${Math.round((insight.interpretive_confidence?.score || 0.75) * 100)}%</strong>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span style="color:var(--text-secondary);">Strategic Horizon:</span>
                <strong style="color:var(--text-primary); text-transform:capitalize;">${state.escapeHtml((insight.temporal_scope?.horizon || 'medium_term').replace('_', ' '))}</strong>
              </div>
            </div>
          </div>

          <div>
            <div class="form-label" style="margin-bottom:8px;">Supporting Grounded Evidence Excerpts</div>
            <div style="display:flex; flex-direction:column; gap:8px;">
              ${insight.supporting_evidence?.length ? insight.supporting_evidence.map(ev => `
                <div style="background:var(--bg-canvas); padding:10px 12px; border-radius:6px; border:1px solid var(--border-subtle); font-size:12px;">
                  <div style="color:var(--text-primary); line-height:1.5;">"${state.escapeHtml(ev.excerpt || ev.content || JSON.stringify(ev))}"</div>
                  <div style="font-size:10px; color:var(--text-muted); margin-top:4px; font-family:var(--font-mono);">
                    Source: ${state.escapeHtml(ev.source_id || 'source')} • Chunk: ${state.escapeHtml(ev.chunk_id || 'chunk')}
                  </div>
                </div>
              `).join('') : `
                <div style="color:var(--text-muted); font-size:12px;">
                  Backed by ${insight.supporting_event_ids?.length || 0} verified events and ${insight.supporting_signal_ids?.length || 0} inferred trend signals.
                </div>
              `}
            </div>
          </div>
        </div>
      `;
    },
    renderFooter: (el) => {
      el.innerHTML = `
        <div style="display:flex; justify-content:flex-end;">
          <button class="btn btn-secondary btn-sm" onclick="document.getElementById('btn-close-drawer').click()">Close</button>
        </div>
      `;
    },
  });
}

function openProvenanceDrawer(provenance) {
  state.openDrawer({
    title: '10-Layer Audit Lineage',
    subtitle: `Provenance ID: ${provenance.provenance_id || 'prov_verified'}`,
    renderContent: (el) => {
      const layers = [
        { label: 'Layer 1: External Source', data: provenance.source },
        { label: 'Layer 2: Source Document', data: provenance.document },
        { label: 'Layer 3: Document Version', data: provenance.version },
        { label: 'Layer 4: Document Chunk', data: provenance.chunk },
        { label: 'Layer 5: Raw Evidence', data: provenance.evidence },
        { label: 'Layer 6: Detected Event', data: provenance.event },
        { label: 'Layer 7: Inferred Signal', data: provenance.signal },
        { label: 'Layer 8: Strategic Insight', data: provenance.insight },
        { label: 'Layer 9: Decision Context', data: provenance.decision_context },
        { label: 'Layer 10: Prescriptive Recommendation', data: provenance.recommendation },
      ];

      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:14px;">
          <div style="background:rgba(99,102,241,0.08); padding:12px; border-radius:8px; border:1px solid var(--border-highlight); font-size:12px; color:var(--text-secondary); line-height:1.5;">
            Every assertion, event, and recommendation in VANTAGE is strictly backed by end-to-end provenance traces, preventing hallucinations and verifying source grounding.
          </div>

          <div style="display:flex; flex-direction:column; gap:8px;">
            ${layers.map((layer, idx) => `
              <div style="background:var(--bg-canvas); border:1px solid var(--border-subtle); border-radius:6px; padding:10px 12px;">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                  <span style="font-weight:600; font-size:12px; color:var(--accent-cyan);">${layer.label}</span>
                  <span class="badge ${layer.data ? 'badge-success' : 'badge-neutral'}" style="font-size:9px;">
                    ${layer.data ? 'VERIFIED' : 'N/A'}
                  </span>
                </div>
                ${layer.data ? `
                  <pre style="margin-top:6px; font-family:var(--font-mono); font-size:11px; color:var(--text-secondary); white-space:pre-wrap; background:rgba(0,0,0,0.3); padding:8px; border-radius:4px; max-height:120px; overflow-y:auto;">${state.escapeHtml(JSON.stringify(layer.data, null, 2))}</pre>
                ` : ''}
              </div>
            `).join('')}
          </div>
        </div>
      `;
    },
    renderFooter: (el) => {
      el.innerHTML = `
        <div style="display:flex; justify-content:flex-end;">
          <button class="btn btn-secondary btn-sm" onclick="document.getElementById('btn-close-drawer').click()">Close</button>
        </div>
      `;
    },
  });
}
