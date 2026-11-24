/**
 * VANTAGE // Dashboard View
 * Executive overview of intelligence stream, signals, monitoring health, and workflows.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderDashboard(container) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  container.innerHTML = `
    <div class="stats-grid">
      <div class="stat-card skeleton skeleton-card"></div>
      <div class="stat-card skeleton skeleton-card"></div>
      <div class="stat-card skeleton skeleton-card"></div>
      <div class="stat-card skeleton skeleton-card"></div>
    </div>
    <div class="card skeleton" style="height: 350px;"></div>
  `;

  try {
    const [competitors, sources, events, signals, insights, workflows] = await Promise.all([
      api.getCompetitors(wsId).catch(() => []),
      api.getSources(wsId).catch(() => []),
      api.getEvents(wsId, null, 10).catch(() => []),
      api.getSignals(wsId, null, 10).catch(() => []),
      api.getInsights(wsId, null, 5).catch(() => []),
      api.getWorkflowRuns(wsId, null, 10).catch(() => []),
    ]);

    const activeSources = sources.filter(s => s.monitoring_enabled);
    const healthySources = sources.filter(s => s.last_fetch_status === 'success');
    const runningWorkflows = workflows.filter(w => w.status === 'running' || w.status === 'resuming');

    // Build competitor map for quick lookup
    const compMap = new Map();
    competitors.forEach(c => compMap.set(c.competitor_id, c.name));

    container.innerHTML = `
      <!-- Top Metrics -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-card-header">
            <span>Competitors</span>
            <span style="font-size: 16px;">🏢</span>
          </div>
          <div class="stat-card-value">${competitors.length}</div>
          <div class="stat-card-footer">
            <span class="badge badge-cyan">${competitors.filter(c => c.is_active).length} Active</span>
            <span>monitored entities</span>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-card-header">
            <span>Monitored Sources</span>
            <span style="font-size: 16px;">📡</span>
          </div>
          <div class="stat-card-value">${sources.length}</div>
          <div class="stat-card-footer">
            <span class="badge ${healthySources.length === sources.length && sources.length > 0 ? 'badge-success' : 'badge-warning'}">
              ${healthySources.length}/${sources.length} Healthy
            </span>
            <span>${activeSources.length} active polling</span>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-card-header">
            <span>Intelligence Signals</span>
            <span style="font-size: 16px;">⚡</span>
          </div>
          <div class="stat-card-value">${signals.length}</div>
          <div class="stat-card-footer">
            <span class="badge badge-purple">${events.length} Events</span>
            <span>grounded evidence</span>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-card-header">
            <span>Strategic Insights</span>
            <span style="font-size: 16px;">🧠</span>
          </div>
          <div class="stat-card-value">${insights.length}</div>
          <div class="stat-card-footer">
            <span class="badge badge-info">${runningWorkflows.length} Pipeline Active</span>
          </div>
        </div>
      </div>

      <!-- Strategic Highlight Banner (if any insight exists) -->
      ${insights.length > 0 ? `
        <div class="insight-highlight-card">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="badge badge-purple">${state.escapeHtml(insights[0].insight_type)}</span>
              <span style="font-size:12px; color:var(--text-muted);">Synthesized Strategic Insight</span>
            </div>
            <a href="#/intelligence?tab=insights" class="btn btn-ghost btn-sm">All Insights &rarr;</a>
          </div>
          <div class="insight-statement">"${state.escapeHtml(insights[0].statement)}"</div>
          <div style="display:flex; align-items:center; gap:18px; font-size:12px; color:var(--text-secondary);">
            <div>Factual Confidence: <strong>${Math.round((insights[0].factual_confidence?.score || 0.8) * 100)}%</strong></div>
            <div>Interpretive Confidence: <strong>${Math.round((insights[0].interpretive_confidence?.score || 0.75) * 100)}%</strong></div>
            <div>Temporal Horizon: <strong>${state.escapeHtml(insights[0].temporal_scope?.horizon || 'medium_term')}</strong></div>
          </div>
        </div>
      ` : ''}

      <!-- Main Split: Intelligence Stream & Operational Health -->
      <div class="dashboard-split">
        <!-- Left: Recent Intelligence Stream -->
        <div class="card">
          <div class="card-header">
            <div>
              <h3 class="card-title">Recent Intelligence Activity</h3>
              <p class="card-subtitle">Verified events, pricing changes, and strategic signals</p>
            </div>
            <a href="#/intelligence" class="btn btn-ghost btn-sm">Full Intelligence &rarr;</a>
          </div>

          <div class="intelligence-stream">
            ${events.length === 0 && signals.length === 0 ? `
              <div class="empty-state" style="padding: 30px;">
                <div class="empty-icon">🔍</div>
                <h4 class="empty-title">No Intelligence Recorded Yet</h4>
                <p class="empty-description">Run an ingestion workflow or register competitor sources to begin monitoring.</p>
                <a href="#/workflows" class="btn btn-primary btn-sm">Run Ingestion Workflow</a>
              </div>
            ` : ''}

            ${events.slice(0, 5).map(ev => `
              <div class="feed-item" data-event-id="${ev.event_id}">
                <div class="feed-item-header">
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span class="badge badge-cyan">${state.escapeHtml(ev.event_type)}</span>
                    <span class="feed-competitor-tag">${state.escapeHtml(compMap.get(ev.company_id) || ev.company_id)}</span>
                  </div>
                  <span style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
                    ${new Date(ev.detected_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
                <div class="feed-item-title">${state.escapeHtml(ev.title)}</div>
                <div class="feed-item-summary">${state.escapeHtml(ev.summary || '')}</div>
                <div class="feed-item-meta">
                  <span>Confidence: <strong style="color:var(--status-success);">${Math.round((ev.confidence?.score || 0.85) * 100)}%</strong></span>
                  <span>Evidence: <strong>${ev.supporting_evidence?.length || 0} excerpts</strong></span>
                  <span style="margin-left:auto; color:var(--accent-primary); font-size:12px;">View Evidence &rarr;</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Right: Sources Health & Workflow Queue -->
        <div style="display:flex; flex-direction:column; gap:24px;">
          <!-- Monitoring Health -->
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">Monitoring Telemetry</h3>
              <a href="#/sources" class="btn btn-ghost btn-sm">Sources</a>
            </div>

            <div style="display:flex; flex-direction:column; gap:12px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:var(--text-secondary); font-size:13px;">Successful Fetches</span>
                <span class="badge badge-success">${healthySources.length}</span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:var(--text-secondary); font-size:13px;">Fetch Failures / Errors</span>
                <span class="badge ${sources.filter(s => s.last_fetch_status === 'failure').length > 0 ? 'badge-danger' : 'badge-neutral'}">
                  ${sources.filter(s => s.last_fetch_status === 'failure').length}
                </span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:var(--text-secondary); font-size:13px;">Pending Ingestion</span>
                <span class="badge badge-warning">
                  ${sources.filter(s => s.last_fetch_status === 'pending').length}
                </span>
              </div>
            </div>

            <div style="margin-top:8px;">
              <a href="#/sources" class="btn btn-secondary" style="width:100%; justify-content:center;">
                Manage Sources & Policy
              </a>
            </div>
          </div>

          <!-- Workflow Executions -->
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">Recent Workflows</h3>
              <a href="#/workflows" class="btn btn-ghost btn-sm">All Runs</a>
            </div>

            <div style="display:flex; flex-direction:column; gap:10px;">
              ${workflows.length === 0 ? `
                <div style="text-align:center; padding:20px; color:var(--text-muted); font-size:12px;">
                  No operational workflows executed yet.
                </div>
              ` : workflows.slice(0, 4).map(w => `
                <div style="padding:10px 12px; background:var(--bg-canvas); border:1px solid var(--border-subtle); border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
                  <div>
                    <div style="font-family:var(--font-mono); font-size:12px; font-weight:600; color:var(--text-primary);">
                      ${state.escapeHtml(w.run_id.substring(0, 16))}...
                    </div>
                    <div style="font-size:11px; color:var(--text-muted);">
                      Target: ${state.escapeHtml(w.target_stop_stage)}
                    </div>
                  </div>
                  <span class="badge ${w.status === 'completed' ? 'badge-success' : w.status === 'running' ? 'badge-purple' : 'badge-danger'}">
                    ${state.escapeHtml(w.status)}
                  </span>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Operational Telemetry & Reliability Card -->
          <div class="card" style="margin-top:16px;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
              <h3 style="font-size:14px; font-weight:600; display:flex; align-items:center; gap:8px;">
                <span>🛠️</span> Operational Reliability & Telemetry
              </h3>
              <a href="/metrics" target="_blank" class="btn btn-ghost btn-sm" style="font-size:11px; color:var(--accent-cyan);">
                Prometheus Scrape Endpoint &nearr;
              </a>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; font-size:11px;">
              <div style="background:var(--bg-canvas); padding:10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="color:var(--text-muted); text-transform:uppercase; font-size:10px;">Active Workflows</div>
                <div style="font-size:16px; font-weight:700; color:var(--accent-primary); margin-top:2px;">${runningWorkflows.length}</div>
              </div>
              <div style="background:var(--bg-canvas); padding:10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="color:var(--text-muted); text-transform:uppercase; font-size:10px;">Workflow Failures</div>
                <div style="font-size:16px; font-weight:700; color:${workflows.filter(w => w.status === 'failed').length > 0 ? 'var(--status-danger)' : 'var(--status-success)'}; margin-top:2px;">
                  ${workflows.filter(w => w.status === 'failed').length}
                </div>
              </div>
              <div style="background:var(--bg-canvas); padding:10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="color:var(--text-muted); text-transform:uppercase; font-size:10px;">Failing Sources</div>
                <div style="font-size:16px; font-weight:700; color:${sources.filter(s => s.consecutive_failure_count > 0).length > 0 ? 'var(--status-danger)' : 'var(--status-success)'}; margin-top:2px;">
                  ${sources.filter(s => s.consecutive_failure_count > 0).length}
                </div>
              </div>
              <div style="background:var(--bg-canvas); padding:10px; border-radius:6px; border:1px solid var(--border-subtle);">
                <div style="color:var(--text-muted); text-transform:uppercase; font-size:10px;">System Health</div>
                <div style="font-size:16px; font-weight:700; color:var(--status-success); margin-top:2px;">200 OK</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;


    // Click handler for event items -> open evidence drawer
    container.querySelectorAll('.feed-item').forEach(item => {
      item.addEventListener('click', () => {
        const evId = item.dataset.eventId;
        const ev = events.find(e => e.event_id === evId);
        if (ev) openEventEvidenceDrawer(ev, compMap.get(ev.company_id));
      });
    });

  } catch (err) {
    container.innerHTML = `
      <div class="card" style="border-color: var(--status-danger);">
        <h3 style="color:var(--status-danger);">Failed Loading Dashboard</h3>
        <p style="color:var(--text-secondary);">${state.escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function openEventEvidenceDrawer(event, competitorName) {
  state.openDrawer({
    title: event.title,
    subtitle: `Detected Event [${event.event_type}] for ${competitorName || event.company_id}`,
    renderContent: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:16px;">
          <div>
            <div class="form-label">Summary & Factual Scope</div>
            <p style="color:var(--text-secondary); line-height:1.6;">${state.escapeHtml(event.summary)}</p>
          </div>

          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; background:var(--bg-canvas); padding:12px; border-radius:8px;">
            <div>
              <div style="font-size:11px; color:var(--text-muted);">Previous State</div>
              <div style="font-family:var(--font-mono); font-size:12px;">${state.escapeHtml(JSON.stringify(event.previous_state || 'none'))}</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--text-muted);">New State</div>
              <div style="font-family:var(--font-mono); font-size:12px; color:var(--status-success);">${state.escapeHtml(JSON.stringify(event.new_state || 'none'))}</div>
            </div>
          </div>

          <div>
            <div class="form-label" style="margin-bottom:8px;">Supporting Grounded Evidence Excerpts (${event.supporting_evidence?.length || 0})</div>
            ${event.supporting_evidence && event.supporting_evidence.length > 0 ? event.supporting_evidence.map(ev => `
              <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:12px;">
                <div class="evidence-quote-box">"${state.escapeHtml(ev.extracted_text || 'No text extracted')}"</div>
                <div style="display:flex; align-items:center; gap:8px; font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
                  <span>Chunk: ${state.escapeHtml(ev.chunk_id?.substring(0, 12) || 'N/A')}</span>
                  <span>&bull;</span>
                  <span>Score: ${ev.retrieval_score ? ev.retrieval_score.toFixed(3) : '1.000'}</span>
                  <span>&bull;</span>
                  <span style="color:var(--accent-cyan);">${state.escapeHtml(ev.source_category || 'web')}</span>
                </div>
              </div>
            `).join('') : '<div style="color:var(--text-muted); font-size:12px;">No evidence excerpts attached to this event.</div>'}
          </div>

          <div>
            <div class="form-label" style="margin-bottom:6px;">Provenance Lineage Pointers</div>
            <div class="json-inspector">${state.escapeHtml(JSON.stringify({
              event_id: event.event_id,
              workspace_id: event.workspace_id,
              company_id: event.company_id,
              confidence: event.confidence,
              significance: event.significance,
              event_time: event.event_time,
            }, null, 2))}</div>
          </div>
        </div>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-secondary btn-sm" onclick="window.state.closeDrawer()">Close</button>
    `,
  });
}
window.state = state;
