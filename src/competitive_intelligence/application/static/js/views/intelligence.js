/**
 * VANTAGE // Intelligence View
 * Evidence-grounded Intelligence Events, Inferred Signals, and Strategic Insights.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderIntelligence(container, { query = {} } = {}) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  const activeTab = query.tab || 'events';

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
      <div>
        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Competitive Intelligence</h2>
        <p style="color:var(--text-muted); font-size:13px;">Evidence-grounded taxonomy: Events &rarr; Signals &rarr; Strategic Insights</p>
      </div>
      <div style="display:flex; align-items:center; gap:12px;">
        <select class="form-control" id="intel-filter-competitor" style="width:180px; padding:6px 12px; font-size:12px;">
          <option value="">All Competitors</option>
        </select>
        <input type="text" class="form-control" id="intel-search" placeholder="Search keywords..." style="width:200px; padding:6px 12px; font-size:12px;" />
      </div>
    </div>

    <!-- Sub-tab Navigation -->
    <div class="tab-bar">
      <button class="tab-btn ${activeTab === 'events' ? 'active' : ''}" data-tab="events">
        <span>⚡ Detected Events</span>
        <span class="badge badge-neutral" id="tab-count-events">-</span>
      </button>
      <button class="tab-btn ${activeTab === 'signals' ? 'active' : ''}" data-tab="signals">
        <span>📶 Inferred Signals</span>
        <span class="badge badge-neutral" id="tab-count-signals">-</span>
      </button>
      <button class="tab-btn ${activeTab === 'insights' ? 'active' : ''}" data-tab="insights">
        <span>🧠 Strategic Insights</span>
        <span class="badge badge-neutral" id="tab-count-insights">-</span>
      </button>
    </div>

    <div id="intel-content-area">
      <div class="skeleton skeleton-card"></div>
    </div>
  `;

  // Wire tab switches
  container.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      window.location.hash = `#/intelligence?tab=${tab}`;
    });
  });

  try {
    const [competitors, events, signals, insights] = await Promise.all([
      api.getCompetitors(wsId),
      api.getEvents(wsId, null, 100),
      api.getSignals(wsId, null, 100),
      api.getInsights(wsId, null, 100),
    ]);

    const compMap = new Map();
    competitors.forEach(c => compMap.set(c.competitor_id, c.name));

    // Update counts
    document.getElementById('tab-count-events').textContent = events.length;
    document.getElementById('tab-count-signals').textContent = signals.length;
    document.getElementById('tab-count-insights').textContent = insights.length;

    // Populate filter dropdown
    const compSelect = document.getElementById('intel-filter-competitor');
    if (compSelect) {
      competitors.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.competitor_id;
        opt.textContent = c.name;
        compSelect.appendChild(opt);
      });
    }

    const contentArea = document.getElementById('intel-content-area');
    const searchInput = document.getElementById('intel-search');

    const updateView = () => {
      const selectedComp = compSelect?.value || '';
      const searchQuery = (searchInput?.value || '').toLowerCase().trim();

      if (activeTab === 'events') {
        renderEventsTab(contentArea, events, compMap, selectedComp, searchQuery);
      } else if (activeTab === 'signals') {
        renderSignalsTab(contentArea, signals, compMap, selectedComp, searchQuery, events);
      } else if (activeTab === 'insights') {
        renderInsightsTab(contentArea, insights, compMap, selectedComp, searchQuery);
      }
    };

    compSelect?.addEventListener('change', updateView);
    searchInput?.addEventListener('input', updateView);

    updateView();

  } catch (err) {
    const contentArea = document.getElementById('intel-content-area');
    if (contentArea) {
      contentArea.innerHTML = `
        <div class="card" style="border-color:var(--status-danger);">
          <h3 style="color:var(--status-danger);">Failed Loading Intelligence</h3>
          <p style="color:var(--text-secondary);">${state.escapeHtml(err.message)}</p>
        </div>
      `;
    }
  }
}

function renderEventsTab(container, events, compMap, filterComp, filterSearch) {
  let filtered = events;
  if (filterComp) filtered = filtered.filter(e => e.company_id === filterComp);
  if (filterSearch) {
    filtered = filtered.filter(e =>
      (e.title || '').toLowerCase().includes(filterSearch) ||
      (e.summary || '').toLowerCase().includes(filterSearch) ||
      (e.event_type || '').toLowerCase().includes(filterSearch)
    );
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚡</div>
        <h3 class="empty-title">No Detected Events Match Filter</h3>
        <p class="empty-description">Events represent discrete factual state changes verified against raw source snapshots.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th>Event Type / Title</th>
            <th>Competitor</th>
            <th>Confidence / Significance</th>
            <th>Evidence Provenance</th>
            <th>Detected Date</th>
            <th style="text-align:right;">Actions</th>
          </tr>
        </thead>
        <tbody>
          ${filtered.map(ev => `
            <tr>
              <td>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                  <span class="badge badge-cyan">${state.escapeHtml(ev.event_type)}</span>
                  <span style="font-weight:600; color:var(--text-primary); font-size:13px;">${state.escapeHtml(ev.title)}</span>
                </div>
                <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">${state.escapeHtml(ev.summary || '')}</div>
              </td>
              <td>
                <span style="font-weight:600; color:var(--accent-cyan);">${state.escapeHtml(compMap.get(ev.company_id) || ev.company_id)}</span>
              </td>
              <td>
                <div style="display:flex; align-items:center; gap:6px;">
                  <span class="badge badge-success">${Math.round((ev.confidence?.score || 0.85) * 100)}%</span>
                  <span class="badge badge-purple">${state.escapeHtml(ev.significance?.level || 'moderate')}</span>
                </div>
              </td>
              <td>
                <span class="badge badge-neutral">${ev.supporting_evidence?.length || 0} Excerpts</span>
              </td>
              <td class="mono-cell">
                ${new Date(ev.detected_at).toLocaleDateString()}
              </td>
              <td style="text-align:right;">
                <button class="btn btn-secondary btn-sm btn-inspect-event" data-event-id="${ev.event_id}">
                  Inspect Evidence
                </button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;

  container.querySelectorAll('.btn-inspect-event').forEach(btn => {
    btn.addEventListener('click', () => {
      const evId = btn.dataset.eventId;
      const ev = events.find(e => e.event_id === evId);
      if (ev) openEventDrawer(ev, compMap.get(ev.company_id));
    });
  });
}

function renderSignalsTab(container, signals, compMap, filterComp, filterSearch, allEvents) {
  let filtered = signals;
  if (filterComp) filtered = filtered.filter(s => s.company_id === filterComp);
  if (filterSearch) {
    filtered = filtered.filter(s =>
      (s.title || '').toLowerCase().includes(filterSearch) ||
      (s.summary || '').toLowerCase().includes(filterSearch) ||
      (s.signal_type || '').toLowerCase().includes(filterSearch)
    );
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📶</div>
        <h3 class="empty-title">No Inferred Signals Found</h3>
        <p class="empty-description">Signals represent synthesized multi-event patterns indicating strategic shifts.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th>Signal Type / Title</th>
            <th>Competitor</th>
            <th>Status</th>
            <th>Supporting Events</th>
            <th>Observation Window</th>
            <th style="text-align:right;">Actions</th>
          </tr>
        </thead>
        <tbody>
          ${filtered.map(sig => `
            <tr>
              <td>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                  <span class="badge badge-purple">${state.escapeHtml(sig.signal_type)}</span>
                  <span style="font-weight:600; color:var(--text-primary); font-size:13px;">${state.escapeHtml(sig.title)}</span>
                </div>
                <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">${state.escapeHtml(sig.summary || '')}</div>
              </td>
              <td>
                <span style="font-weight:600; color:var(--accent-cyan);">${state.escapeHtml(compMap.get(sig.company_id) || sig.company_id)}</span>
              </td>
              <td>
                <span class="badge ${sig.status === 'confirmed' ? 'badge-success' : 'badge-warning'}">
                  ${state.escapeHtml(sig.status)}
                </span>
              </td>
              <td>
                <span class="badge badge-neutral">${sig.supporting_event_ids?.length || 0} Events Linked</span>
              </td>
              <td class="mono-cell" style="font-size:11px;">
                ${sig.first_observed_at ? new Date(sig.first_observed_at).toLocaleDateString() : 'N/A'} - ${sig.last_observed_at ? new Date(sig.last_observed_at).toLocaleDateString() : 'now'}
              </td>
              <td style="text-align:right;">
                <button class="btn btn-secondary btn-sm btn-inspect-signal" data-sig-id="${sig.signal_id}">
                  View Pattern
                </button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;

  container.querySelectorAll('.btn-inspect-signal').forEach(btn => {
    btn.addEventListener('click', () => {
      const sId = btn.dataset.sigId;
      const sig = signals.find(s => s.signal_id === sId);
      if (sig) openSignalDrawer(sig, compMap.get(sig.company_id), allEvents);
    });
  });
}

function renderInsightsTab(container, insights, compMap, filterComp, filterSearch) {
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
        <p class="empty-description">Insights synthesize multi-source evidence into high-level implications for decision-makers.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:16px;">
      ${filtered.map(ins => `
        <div class="card" style="border-left:4px solid var(--accent-primary);">
          <div class="card-header">
            <div style="display:flex; align-items:center; gap:10px;">
              <span class="badge badge-purple">${state.escapeHtml(ins.insight_type)}</span>
              <span class="card-title">${state.escapeHtml(ins.title || 'Strategic Synthesis')}</span>
              <span style="font-size:12px; color:var(--accent-cyan); font-weight:600;">${state.escapeHtml(compMap.get(ins.company_id) || ins.company_id)}</span>
            </div>
            <span class="badge ${ins.status === 'active' ? 'badge-success' : 'badge-neutral'}">${state.escapeHtml(ins.status)}</span>
          </div>

          <div class="insight-statement">
            "${state.escapeHtml(ins.statement)}"
          </div>

          <p style="font-size:13px; color:var(--text-secondary); line-height:1.5;">
            ${state.escapeHtml(ins.summary || '')}
          </p>

          <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; padding-top:12px; border-top:1px solid var(--border-subtle);">
            <div style="display:flex; align-items:center; gap:16px; font-size:12px; color:var(--text-muted);">
              <span>Factual Confidence: <strong style="color:var(--text-primary);">${Math.round((ins.factual_confidence?.score || 0.8) * 100)}%</strong></span>
              <span>Interpretive Confidence: <strong style="color:var(--text-primary);">${Math.round((ins.interpretive_confidence?.score || 0.75) * 100)}%</strong></span>
              <span>Horizon: <strong>${state.escapeHtml(ins.temporal_scope?.horizon || 'medium_term')}</strong></span>
            </div>

            <div style="display:flex; align-items:center; gap:8px;">
              <span class="badge badge-neutral">${ins.supporting_event_ids?.length || 0} Events</span>
              <span class="badge badge-neutral">${ins.supporting_signal_ids?.length || 0} Signals</span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function openEventDrawer(event, compName) {
  state.openDrawer({
    title: event.title,
    subtitle: `Detected Event [${event.event_type}] for ${compName || event.company_id}`,
    renderContent: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:18px;">
          <div>
            <div class="form-label">Subject & Event Summary</div>
            <p style="color:var(--text-secondary); line-height:1.6; margin-top:4px;">${state.escapeHtml(event.summary)}</p>
          </div>

          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; background:var(--bg-canvas); padding:12px; border-radius:8px;">
            <div>
              <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Previous State</div>
              <div style="font-family:var(--font-mono); font-size:12px; margin-top:4px;">${state.escapeHtml(JSON.stringify(event.previous_state || 'none'))}</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">New State</div>
              <div style="font-family:var(--font-mono); font-size:12px; color:var(--status-success); margin-top:4px;">${state.escapeHtml(JSON.stringify(event.new_state || 'none'))}</div>
            </div>
          </div>

          <div>
            <div class="form-label" style="margin-bottom:10px;">Grounded Evidence Lineage (${event.supporting_evidence?.length || 0})</div>
            ${event.supporting_evidence && event.supporting_evidence.length > 0 ? event.supporting_evidence.map(ev => `
              <div style="display:flex; flex-direction:column; gap:6px; margin-bottom:12px;">
                <div class="evidence-quote-box">"${state.escapeHtml(ev.extracted_text)}"</div>
                <div style="display:flex; align-items:center; gap:8px; font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
                  <span>Chunk: ${state.escapeHtml(ev.chunk_id?.substring(0, 10))}...</span>
                  <span>&bull;</span>
                  <span>Score: ${ev.retrieval_score ? ev.retrieval_score.toFixed(3) : '1.0'}</span>
                  <span>&bull;</span>
                  <a href="${state.escapeHtml(ev.source_url)}" target="_blank" style="color:var(--accent-cyan); text-decoration:none;">Source URL &nearr;</a>
                </div>
              </div>
            `).join('') : '<div style="color:var(--text-muted); font-size:12px;">No evidence citations attached.</div>'}
          </div>
        </div>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-secondary btn-sm" onclick="window.state.closeDrawer()">Close</button>
    `,
  });
}

function openSignalDrawer(signal, compName, allEvents) {
  const linkedEvents = (allEvents || []).filter(e => signal.supporting_event_ids?.includes(e.event_id));

  state.openDrawer({
    title: signal.title,
    subtitle: `Inferred Pattern [${signal.signal_type}] for ${compName || signal.company_id}`,
    renderContent: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:18px;">
          <div>
            <div class="form-label">Strategic Thesis</div>
            <p style="color:var(--text-secondary); line-height:1.6; margin-top:4px;">${state.escapeHtml(signal.summary)}</p>
          </div>

          <div>
            <div class="form-label" style="margin-bottom:8px;">Linked Supporting Events (${linkedEvents.length})</div>
            <div style="display:flex; flex-direction:column; gap:8px;">
              ${linkedEvents.map(ev => `
                <div style="background:var(--bg-canvas); border:1px solid var(--border-subtle); padding:10px 14px; border-radius:8px;">
                  <div style="display:flex; align-items:center; justify-content:space-between;">
                    <span class="badge badge-cyan" style="font-size:10px;">${state.escapeHtml(ev.event_type)}</span>
                    <span style="font-size:11px; color:var(--text-muted);">${new Date(ev.detected_at).toLocaleDateString()}</span>
                  </div>
                  <div style="font-weight:600; font-size:13px; color:var(--text-primary); margin-top:4px;">${state.escapeHtml(ev.title)}</div>
                </div>
              `).join('')}
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
