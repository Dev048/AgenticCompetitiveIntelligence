/**
 * VANTAGE // Decisions & Decision Support View
 * Strategic Decision Contexts, Candidate Options, and 10-Layer Provenance-backed Recommendations.
 * Enforces strict Human Approval Boundary: Autonomous execution is prohibited.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderDecisions(container) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
      <div>
        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Strategic Decision Support</h2>
        <p style="color:var(--text-muted); font-size:13px;">
          Human-in-the-loop review: Recommendations remain PROPOSED until explicitly approved by an operator
        </p>
      </div>
      <button class="btn btn-primary" id="btn-create-context">
        <span>+</span> Formulate Decision Context
      </button>
    </div>

    <div class="dashboard-split" id="decisions-split-layout">
      <!-- Left: Active Context & Evaluated Options / Recommendations -->
      <div id="decision-detail-area">
        <div class="card skeleton skeleton-card" style="height:350px;"></div>
      </div>

      <!-- Right: Decision Contexts Selector -->
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Decision Contexts</h3>
          </div>
          <div id="contexts-list-container" style="display:flex; flex-direction:column; gap:10px;">
            <div class="skeleton skeleton-text"></div>
            <div class="skeleton skeleton-text"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  try {
    const [contexts, competitors] = await Promise.all([
      api.getDecisionContexts(wsId),
      api.getCompetitors(wsId),
    ]);

    const compMap = new Map();
    competitors.forEach(c => compMap.set(c.competitor_id, c.name));

    document.getElementById('btn-create-context')?.addEventListener('click', () => {
      openCreateContextModal(wsId, competitors, () => renderDecisions(container));
    });

    const contextsListEl = document.getElementById('contexts-list-container');
    const detailAreaEl = document.getElementById('decision-detail-area');

    if (contexts.length === 0) {
      if (detailAreaEl) {
        detailAreaEl.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">⚖️</div>
            <h3 class="empty-title">No Decision Contexts Defined</h3>
            <p class="empty-description">
              Formulate a decision context with your organization's business objectives, constraints, and risk tolerances to evaluate candidate counter-moves.
            </p>
            <button class="btn btn-primary" id="btn-empty-create-ctx">Formulate Decision Context</button>
          </div>
        `;
        detailAreaEl.querySelector('#btn-empty-create-ctx')?.addEventListener('click', () => {
          openCreateContextModal(wsId, competitors, () => renderDecisions(container));
        });
      }
      if (contextsListEl) {
        contextsListEl.innerHTML = '<div style="color:var(--text-muted); font-size:12px;">No contexts found.</div>';
      }
      return;
    }

    // Render Contexts List in Sidebar
    const renderContextsList = (selectedId) => {
      if (!contextsListEl) return;
      contextsListEl.innerHTML = contexts.map(ctx => `
        <div class="feed-item ${ctx.decision_id === selectedId ? 'active' : ''}" data-ctx-id="${ctx.decision_id}" style="padding:12px 14px; border-left:3px solid ${ctx.decision_id === selectedId ? 'var(--accent-primary)' : 'transparent'};">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <span class="badge badge-purple" style="font-size:10px;">${state.escapeHtml(ctx.decision_domain)}</span>
            <span style="font-size:11px; color:var(--text-muted);">${ctx.business_objectives?.length || 0} Objs</span>
          </div>
          <div style="font-weight:600; font-size:13px; color:var(--text-primary); margin-top:4px;">${state.escapeHtml(ctx.title)}</div>
        </div>
      `).join('');

      contextsListEl.querySelectorAll('.feed-item').forEach(item => {
        item.addEventListener('click', () => {
          const cId = item.dataset.ctxId;
          renderContextDetail(cId);
          renderContextsList(cId);
        });
      });
    };

    // Render Context Detail + Options + Recommendations
    const renderContextDetail = async (contextId) => {
      const ctx = contexts.find(c => c.decision_id === contextId) || contexts[0];
      if (!ctx || !detailAreaEl) return;

      detailAreaEl.innerHTML = '<div class="card skeleton skeleton-card" style="height:400px;"></div>';

      const [options, recommendations] = await Promise.all([
        api.getDecisionOptions(wsId, ctx.decision_id).catch(() => []),
        api.getRecommendations(wsId, ctx.decision_id).catch(() => []),
      ]);

      detailAreaEl.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px;">
          <!-- Context Header Card -->
          <div class="card">
            <div class="card-header">
              <div>
                <span class="badge badge-purple" style="margin-bottom:6px;">${state.escapeHtml(ctx.decision_domain)}</span>
                <h3 class="card-title">${state.escapeHtml(ctx.title)}</h3>
                <p class="card-subtitle">${state.escapeHtml(ctx.current_state_summary || 'No current state summary provided.')}</p>
              </div>
              <div style="text-align:right;">
                <span class="badge badge-cyan">${state.escapeHtml(ctx.risk_tolerance)}</span>
                <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Horizon: ${state.escapeHtml(ctx.time_horizon)}</div>
              </div>
            </div>

            <!-- Objectives & Constraints Chips -->
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
              <div style="background:var(--bg-canvas); padding:14px; border-radius:8px; border:1px solid var(--border-subtle);">
                <div class="form-label" style="margin-bottom:8px;">Business Objectives (${ctx.business_objectives?.length || 0})</div>
                <div style="display:flex; flex-direction:column; gap:6px;">
                  ${ctx.business_objectives?.map(o => `
                    <div style="font-size:12px; display:flex; justify-content:space-between;">
                      <span style="font-weight:600; color:var(--text-primary);">${state.escapeHtml(o.name)} [${state.escapeHtml(o.dimension)}]</span>
                      <span style="color:var(--accent-primary); font-family:var(--font-mono);">${o.priority_weight ? Math.round(o.priority_weight * 100) + '%' : ''}</span>
                    </div>
                  `).join('') || '<div style="color:var(--text-muted); font-size:12px;">No objectives specified.</div>'}
                </div>
              </div>

              <div style="background:var(--bg-canvas); padding:14px; border-radius:8px; border:1px solid var(--border-subtle);">
                <div class="form-label" style="margin-bottom:8px;">Operational Constraints (${ctx.constraints?.length || 0})</div>
                <div style="display:flex; flex-direction:column; gap:6px;">
                  ${ctx.constraints?.map(c => `
                    <div style="font-size:12px; display:flex; justify-content:space-between;">
                      <span style="color:var(--text-secondary);">${state.escapeHtml(c.description)}</span>
                      <span class="badge ${c.is_hard_constraint ? 'badge-danger' : 'badge-neutral'}" style="font-size:9px;">
                        ${c.is_hard_constraint ? 'HARD' : 'SOFT'}
                      </span>
                    </div>
                  `).join('') || '<div style="color:var(--text-muted); font-size:12px;">No constraints specified.</div>'}
                </div>
              </div>
            </div>
          </div>

          <!-- Recommendations Section -->
          <div>
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
              <h3 style="font-family:var(--font-display); font-size:16px; font-weight:700;">Proposed Actionable Recommendations (${recommendations.length})</h3>
              <span class="badge badge-warning" style="text-transform:none;">Human approval required</span>
            </div>

            <div style="display:flex; flex-direction:column; gap:16px;">
              ${recommendations.length === 0 ? `
                <div class="card" style="text-align:center; padding:32px;">
                  <div style="color:var(--text-muted); font-size:13px;">
                    No recommendations formulated for this context yet. Run a workflow with <strong>Decision Support</strong> enabled.
                  </div>
                  <div style="margin-top:12px;">
                    <a href="#/workflows" class="btn btn-secondary btn-sm">Dispatch Decision Workflow</a>
                  </div>
                </div>
              ` : recommendations.map(rec => `
                <div class="card" style="border-left:4px solid ${rec.status === 'accepted' ? 'var(--status-success)' : rec.status === 'rejected' ? 'var(--status-danger)' : 'var(--accent-primary)'};">
                  <div class="card-header">
                    <div>
                      <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                        <span class="badge ${rec.urgency === 'urgent' ? 'badge-danger' : 'badge-cyan'}">${state.escapeHtml(rec.urgency)}</span>
                        <span class="badge ${rec.status === 'accepted' ? 'badge-success' : rec.status === 'rejected' ? 'badge-danger' : 'badge-warning'}">
                          STATUS: ${state.escapeHtml(rec.status.toUpperCase())}
                        </span>
                      </div>
                      <h4 style="font-family:var(--font-display); font-size:16px; font-weight:700; color:var(--text-primary);">${state.escapeHtml(rec.title)}</h4>
                    </div>
                    <button class="btn btn-secondary btn-sm btn-view-provenance" data-rec-id="${rec.recommendation_id}">
                      🔍 10-Layer Lineage
                    </button>
                  </div>

                  <p style="font-size:13px; color:var(--text-secondary); line-height:1.6;">${state.escapeHtml(rec.rationale)}</p>

                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; background:var(--bg-canvas); padding:14px; border-radius:8px;">
                    <div>
                      <div class="form-label" style="font-size:11px; margin-bottom:4px;">Expected Strategic Benefits</div>
                      <ul style="padding-left:16px; font-size:12px; color:var(--status-success); line-height:1.5;">
                        ${rec.expected_benefits?.map(b => `<li>${state.escapeHtml(b)}</li>`).join('') || '<li>Standard competitive parity</li>'}
                      </ul>
                    </div>
                    <div>
                      <div class="form-label" style="font-size:11px; margin-bottom:4px;">Trade-offs & Resource Demands</div>
                      <ul style="padding-left:16px; font-size:12px; color:var(--status-warning); line-height:1.5;">
                        ${rec.tradeoffs?.map(t => `<li>${state.escapeHtml(t)}</li>`).join('') || '<li>Standard operational allocation</li>'}
                      </ul>
                    </div>
                  </div>

                  <!-- Human Approval Action Bar -->
                  <div style="display:flex; align-items:center; justify-content:space-between; padding-top:12px; border-top:1px solid var(--border-subtle); flex-wrap:wrap; gap:12px;">
                    <div style="font-size:12px; color:var(--text-muted);">
                      Confidence: <strong style="color:var(--text-primary);">${Math.round((rec.confidence?.score || 0.85) * 100)}%</strong> • Formulated on ${new Date(rec.generated_at).toLocaleDateString()}
                    </div>

                    ${rec.status === 'proposed' ? `
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
                        Reviewed by human operator: <strong>${rec.status.toUpperCase()}</strong>
                      </div>
                    `}
                  </div>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Evaluated Candidate Options -->
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">Candidate Strategic Options (${options.length})</h3>
            </div>
            <div style="display:flex; flex-direction:column; gap:12px;">
              ${options.map(opt => `
                <div style="background:var(--bg-canvas); border:1px solid var(--border-subtle); border-radius:8px; padding:14px;">
                  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                      <span class="badge badge-purple" style="font-size:10px;">${state.escapeHtml(opt.action_type)}</span>
                      <span style="font-weight:600; font-size:14px; color:var(--text-primary);">${state.escapeHtml(opt.title)}</span>
                    </div>
                    ${opt.estimated_cost_usd ? `<span style="font-family:var(--font-mono); font-size:12px; color:var(--status-warning);">$${opt.estimated_cost_usd.toLocaleString()}</span>` : ''}
                  </div>
                  <p style="font-size:12px; color:var(--text-secondary); line-height:1.5;">${state.escapeHtml(opt.description)}</p>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      `;

      // Wire Provenance Viewer
      detailAreaEl.querySelectorAll('.btn-view-provenance').forEach(btn => {
        btn.addEventListener('click', async () => {
          const recId = btn.dataset.recId;
          btn.disabled = true;
          try {
            const prov = await api.getRecommendationProvenance(wsId, recId);
            openProvenanceDrawer(prov);
          } catch (err) {
            state.showToast(err.message, 'error');
          } finally {
            btn.disabled = false;
          }
        });
      });

      // Wire Human Approval / Rejection
      detailAreaEl.querySelectorAll('.btn-approve-rec').forEach(btn => {
        btn.addEventListener('click', () => {
          const recId = btn.dataset.recId;
          const recTitle = btn.dataset.recTitle;
          confirmStatusReview(wsId, recId, recTitle, 'accepted', () => renderContextDetail(ctx.decision_id));
        });
      });

      detailAreaEl.querySelectorAll('.btn-reject-rec').forEach(btn => {
        btn.addEventListener('click', () => {
          const recId = btn.dataset.recId;
          const recTitle = btn.dataset.recTitle;
          confirmStatusReview(wsId, recId, recTitle, 'rejected', () => renderContextDetail(ctx.decision_id));
        });
      });
    };

    renderContextsList(contexts[0]?.decision_id);
    if (contexts.length > 0) {
      renderContextDetail(contexts[0].decision_id);
    }

  } catch (err) {
    container.innerHTML = `
      <div class="card" style="border-color:var(--status-danger);">
        <h3 style="color:var(--status-danger);">Failed Loading Decision Contexts</h3>
        <p style="color:var(--text-secondary);">${state.escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function confirmStatusReview(workspaceId, recId, recTitle, targetStatus, onSuccess) {
  state.openModal({
    title: targetStatus === 'accepted' ? 'Approve Recommendation' : 'Reject Recommendation',
    renderBody: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:16px;">
          <p style="color:var(--text-primary); font-size:14px;">
            Are you sure you want to mark recommendation <strong>"${state.escapeHtml(recTitle)}"</strong> as <span class="badge ${targetStatus === 'accepted' ? 'badge-success' : 'badge-danger'}">${targetStatus.toUpperCase()}</span>?
          </p>
          <div class="form-group">
            <label class="form-label">Review Audit Notes (Optional)</label>
            <textarea class="form-control" id="review-notes" placeholder="Reasoning or operational next steps..."></textarea>
          </div>
          <p style="font-size:11px; color:var(--text-muted);">
            Note: In accordance with platform governance, this updates the status record but does not trigger autonomous actions.
          </p>
        </div>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-ghost" onclick="window.state.closeModal()">Cancel</button>
      <button class="btn ${targetStatus === 'accepted' ? 'btn-success' : 'btn-danger'}" id="btn-confirm-review">
        Confirm ${targetStatus.toUpperCase()}
      </button>
    `,
  });

  document.getElementById('btn-confirm-review')?.addEventListener('click', async () => {
    const notes = document.getElementById('review-notes')?.value.trim();
    const btn = document.getElementById('btn-confirm-review');
    if (btn) btn.disabled = true;
    try {
      await api.updateRecommendationStatus(workspaceId, recId, targetStatus, notes);
      state.showToast(`Recommendation marked ${targetStatus}`, 'success');
      state.closeModal();
      if (onSuccess) onSuccess();
    } catch (err) {
      state.showToast(err.message, 'error');
      if (btn) btn.disabled = false;
    }
  });
}

function openProvenanceDrawer(prov) {
  state.openDrawer({
    title: '10-Layer Grounded Lineage Tree',
    subtitle: `Grounded Provenance Verification for Recommendation ${prov.recommendation_id?.substring(0, 8)}...`,
    renderContent: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:18px;">
          <p style="font-size:13px; color:var(--text-secondary); line-height:1.5;">
            Every prescriptive recommendation is deterministically chained through 10 provenance layers back to immutable document chunks and origin URLs.
          </p>

          <div class="provenance-tree">
            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">1. Recommendation</div>
                <div class="provenance-node-id">${state.escapeHtml(prov.recommendation_id)}</div>
              </div>
              <span class="badge badge-success">Top Anchor</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">2. Decision Option</div>
                <div class="provenance-node-id">${state.escapeHtml(prov.selected_option?.option_id || 'Option Selected')}</div>
              </div>
              <span class="badge badge-purple">${state.escapeHtml(prov.selected_option?.action_type || 'Prescription')}</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">3. Strategic Insights (${prov.insight_count})</div>
                <div class="provenance-node-id">${prov.insight_ids?.slice(0, 2).join(', ') || 'Linked Insights'}</div>
              </div>
              <span class="badge badge-neutral">${prov.insight_count} Verified</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">4. Inferred Signals (${prov.signal_count})</div>
                <div class="provenance-node-id">${prov.signal_ids?.slice(0, 2).join(', ') || 'Linked Signals'}</div>
              </div>
              <span class="badge badge-neutral">${prov.signal_count} Verified</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">5. Detected Events (${prov.event_count})</div>
                <div class="provenance-node-id">${prov.event_ids?.slice(0, 2).join(', ') || 'Discrete Events'}</div>
              </div>
              <span class="badge badge-neutral">${prov.event_count} Verified</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">6. Raw Evidence Citations (${prov.evidence_count})</div>
                <div class="provenance-node-id">${prov.evidence_ids?.slice(0, 2).join(', ') || 'Citations'}</div>
              </div>
              <span class="badge badge-neutral">${prov.evidence_count} Excerpts</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">7. Document Chunks (${prov.chunk_count})</div>
                <div class="provenance-node-id">${prov.chunk_ids?.slice(0, 2).join(', ') || 'Vector Chunks'}</div>
              </div>
              <span class="badge badge-neutral">${prov.chunk_count} Chunks</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">8. Document Versions (${prov.version_count})</div>
                <div class="provenance-node-id">${prov.version_ids?.slice(0, 2).join(', ') || 'Content Hashes'}</div>
              </div>
              <span class="badge badge-neutral">${prov.version_count} Snapshots</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">9. Canonical Documents (${prov.document_count})</div>
                <div class="provenance-node-id">${prov.document_ids?.slice(0, 2).join(', ') || 'Documents'}</div>
              </div>
              <span class="badge badge-neutral">${prov.document_count} Docs</span>
            </div>

            <div class="provenance-node">
              <div>
                <div class="provenance-node-layer">10. Monitored Sources (${prov.source_count})</div>
                <div class="provenance-node-id">${prov.source_ids?.slice(0, 2).join(', ') || 'Origin Source'}</div>
              </div>
              <span class="badge badge-cyan">Origin Root</span>
            </div>
          </div>
        </div>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-secondary btn-sm" onclick="window.state.closeDrawer()">Close</button>
    `,
  });
}

function openCreateContextModal(workspaceId, competitors, onSuccess) {
  state.openModal({
    title: 'Formulate Decision Context',
    renderBody: (el) => {
      el.innerHTML = `
        <form id="form-create-decision" style="display:flex; flex-direction:column; gap:16px;">
          <div class="form-group">
            <label class="form-label">Context Title *</label>
            <input type="text" class="form-control" id="dec-title" placeholder="e.g. Q1 SaaS Pricing Strategy Counter-Move" required />
          </div>

          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div class="form-group">
              <label class="form-label">Decision Domain</label>
              <select class="form-control" id="dec-domain">
                <option value="pricing">Pricing Strategy</option>
                <option value="product">Product & Features</option>
                <option value="gtm_marketing">GTM & Marketing</option>
                <option value="hiring_talent">Hiring & Talent</option>
                <option value="technology_infrastructure">Technology Infrastructure</option>
                <option value="risk_compliance">Risk & Compliance</option>
                <option value="general_strategy">General Strategy</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">Risk Tolerance</label>
              <select class="form-control" id="dec-risk">
                <option value="very_low_conservative">Conservative (Very Low)</option>
                <option value="moderate" selected>Moderate</option>
                <option value="high_aggressive">Aggressive (High)</option>
              </select>
            </div>
          </div>

          <div class="form-group">
            <label class="form-label">Primary Business Objective *</label>
            <input type="text" class="form-control" id="dec-obj-name" placeholder="Objective name (e.g. Protect Enterprise ARR)" required />
            <input type="text" class="form-control" id="dec-obj-target" placeholder="Target description (e.g. Limit churn under 2% during competitor price drop)" style="margin-top:6px;" required />
          </div>

          <div class="form-group">
            <label class="form-label">Operational Constraint (Optional)</label>
            <input type="text" class="form-control" id="dec-constraint-desc" placeholder="Constraint (e.g. Budget ceiling $50,000 for discount matching)" />
          </div>

          <div class="form-group">
            <label class="form-label">Current Situation Summary</label>
            <textarea class="form-control" id="dec-summary" placeholder="Executive briefing on the competitive trigger..."></textarea>
          </div>
        </form>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-ghost" onclick="window.state.closeModal()">Cancel</button>
      <button class="btn btn-primary" id="btn-submit-context">Create Context</button>
    `,
  });

  document.getElementById('btn-submit-context')?.addEventListener('click', async () => {
    const title = document.getElementById('dec-title')?.value.trim();
    const domain = document.getElementById('dec-domain')?.value;
    const risk = document.getElementById('dec-risk')?.value;
    const objName = document.getElementById('dec-obj-name')?.value.trim();
    const objTarget = document.getElementById('dec-obj-target')?.value.trim();
    const constraintDesc = document.getElementById('dec-constraint-desc')?.value.trim();
    const summary = document.getElementById('dec-summary')?.value.trim();

    if (!title || !objName || !objTarget) {
      state.showToast('Please fill title and business objective fields', 'warning');
      return;
    }

    const btn = document.getElementById('btn-submit-context');
    if (btn) btn.disabled = true;

    try {
      const payload = {
        title,
        decision_domain: domain,
        competitor_ids: competitors.map(c => c.competitor_id),
        risk_tolerance: risk,
        time_horizon: 'medium_term',
        current_state_summary: summary,
        business_objectives: [
          {
            name: objName,
            dimension: domain === 'pricing' ? 'revenue' : 'market_share',
            target_description: objTarget,
            priority_weight: 1.0,
          },
        ],
        constraints: constraintDesc ? [
          {
            constraint_type: 'budget',
            description: constraintDesc,
            is_hard_constraint: true,
          },
        ] : [],
      };

      await api.createDecisionContext(workspaceId, payload);
      state.showToast(`Decision context '${title}' created`, 'success');
      state.closeModal();
      if (onSuccess) onSuccess();
    } catch (err) {
      state.showToast(err.message, 'error');
      if (btn) btn.disabled = false;
    }
  });
}
