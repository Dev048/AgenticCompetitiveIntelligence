/**
 * VANTAGE // Competitors View
 * Tracked competitor management, registration, and competitor-scoped intelligence history.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderCompetitors(container) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between;">
      <div>
        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Tracked Competitors</h2>
        <p style="color:var(--text-muted); font-size:13px;">Manage market rivals and configure their monitoring perimeter</p>
      </div>
      <button class="btn btn-primary" id="btn-add-competitor">
        <span>+</span> Register Competitor
      </button>
    </div>

    <div class="competitors-grid" id="competitors-grid-container">
      <div class="stat-card skeleton skeleton-card"></div>
      <div class="stat-card skeleton skeleton-card"></div>
      <div class="stat-card skeleton skeleton-card"></div>
    </div>
  `;

  document.getElementById('btn-add-competitor')?.addEventListener('click', () => {
    openAddCompetitorModal(wsId, () => renderCompetitors(container));
  });

  try {
    const [competitors, sources, events] = await Promise.all([
      api.getCompetitors(wsId),
      api.getSources(wsId).catch(() => []),
      api.getEvents(wsId, null, 200).catch(() => []),
    ]);

    const grid = document.getElementById('competitors-grid-container');
    if (!grid) return;

    if (competitors.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">🏢</div>
          <h3 class="empty-title">No Competitors Tracked</h3>
          <p class="empty-description">
            Register your first competitor to monitor their pricing tiers, product releases, executive hiring, and regulatory filings.
          </p>
          <button class="btn btn-primary" id="btn-empty-add-comp">Register Competitor</button>
        </div>
      `;
      grid.querySelector('#btn-empty-add-comp')?.addEventListener('click', () => {
        openAddCompetitorModal(wsId, () => renderCompetitors(container));
      });
      return;
    }

    grid.innerHTML = competitors.map(comp => {
      const compSources = sources.filter(s => s.company_id === comp.competitor_id);
      const compEvents = events.filter(e => e.company_id === comp.competitor_id);

      return `
        <div class="competitor-card" data-comp-id="${comp.competitor_id}">
          <div class="competitor-header">
            <div>
              <div class="competitor-name">${state.escapeHtml(comp.name)}</div>
              <div class="competitor-domain">${state.escapeHtml(comp.domain || 'no domain registered')}</div>
            </div>
            <span class="badge ${comp.is_active ? 'badge-success' : 'badge-neutral'}">
              ${comp.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>

          <p style="font-size:13px; color:var(--text-secondary); line-height:1.5; min-height:38px;">
            ${state.escapeHtml(comp.description || 'No description provided.')}
          </p>

          <div class="competitor-counts">
            <div>
              <div class="count-label">Sources</div>
              <div class="count-val" style="color:var(--accent-cyan);">${compSources.length}</div>
            </div>
            <div>
              <div class="count-label">Events</div>
              <div class="count-val" style="color:var(--accent-primary);">${compEvents.length}</div>
            </div>
            <div>
              <div class="count-label">Status</div>
              <div class="count-val" style="font-size:12px; color:${comp.is_active ? 'var(--status-success)' : 'var(--text-muted)'};">
                ${comp.is_active ? 'TRACKED' : 'PAUSED'}
              </div>
            </div>
          </div>

          <div style="display:flex; align-items:center; justify-content:space-between; margin-top:4px;">
            <button class="btn btn-secondary btn-sm btn-inspect-comp" data-comp-id="${comp.competitor_id}">
              View Perimeter
            </button>
            <button class="btn btn-ghost btn-sm btn-toggle-active" data-comp-id="${comp.competitor_id}" data-active="${comp.is_active}">
              ${comp.is_active ? 'Pause' : 'Activate'}
            </button>
          </div>
        </div>
      `;
    }).join('');

    // Wire Card Click / Inspect Handlers
    grid.querySelectorAll('.btn-inspect-comp').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const compId = btn.dataset.compId;
        const comp = competitors.find(c => c.competitor_id === compId);
        if (comp) {
          const compSources = sources.filter(s => s.company_id === compId);
          const compEvents = events.filter(e => e.company_id === compId);
          openCompetitorDrawer(comp, compSources, compEvents, wsId, () => renderCompetitors(container));
        }
      });
    });

    grid.querySelectorAll('.btn-toggle-active').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const compId = btn.dataset.compId;
        const currentActive = btn.dataset.active === 'true';
        btn.disabled = true;
        try {
          await api.updateCompetitor(wsId, compId, { is_active: !currentActive });
          state.showToast(`Competitor ${!currentActive ? 'activated' : 'paused'} successfully`, 'success');
          renderCompetitors(container);
        } catch (err) {
          state.showToast(err.message, 'error');
          btn.disabled = false;
        }
      });
    });

  } catch (err) {
    container.innerHTML = `
      <div class="card" style="border-color:var(--status-danger);">
        <h3 style="color:var(--status-danger);">Error Loading Competitors</h3>
        <p style="color:var(--text-secondary);">${state.escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function openAddCompetitorModal(workspaceId, onSuccess) {
  state.openModal({
    title: 'Register Tracked Competitor',
    renderBody: (el) => {
      el.innerHTML = `
        <form id="form-register-competitor" style="display:flex; flex-direction:column; gap:16px;">
          <div class="form-group">
            <label class="form-label">Competitor Name *</label>
            <input type="text" class="form-control" id="comp-name" placeholder="e.g. Stripe, Snowflake, Datadog" required />
          </div>
          <div class="form-group">
            <label class="form-label">Domain Name</label>
            <input type="text" class="form-control" id="comp-domain" placeholder="e.g. stripe.com" />
            <div class="form-help">Canonical primary domain for evidence matching</div>
          </div>
          <div class="form-group">
            <label class="form-label">Strategic Description</label>
            <textarea class="form-control" id="comp-desc" placeholder="Core market focus, primary offerings, competitive threats..."></textarea>
          </div>
        </form>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-ghost" onclick="window.state.closeModal()">Cancel</button>
      <button class="btn btn-primary" id="btn-submit-competitor">Register</button>
    `,
  });

  document.getElementById('btn-submit-competitor')?.addEventListener('click', async () => {
    const name = document.getElementById('comp-name')?.value.trim();
    const domain = document.getElementById('comp-domain')?.value.trim();
    const description = document.getElementById('comp-desc')?.value.trim();

    if (!name) {
      state.showToast('Competitor name is required', 'warning');
      return;
    }

    const submitBtn = document.getElementById('btn-submit-competitor');
    if (submitBtn) submitBtn.disabled = true;

    try {
      await api.createCompetitor(workspaceId, { name, domain, description });
      state.showToast(`Competitor '${name}' registered successfully`, 'success');
      state.closeModal();
      if (onSuccess) onSuccess();
    } catch (err) {
      state.showToast(err.message, 'error');
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

function openCompetitorDrawer(comp, sources, events, workspaceId, onRefresh) {
  state.openDrawer({
    title: comp.name,
    subtitle: `${comp.domain || 'Domain not specified'} • Registered ${new Date(comp.created_at).toLocaleDateString()}`,
    renderContent: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px;">
          <div>
            <div class="form-label">Strategic Overview</div>
            <p style="color:var(--text-secondary); line-height:1.6; font-size:13px;">
              ${state.escapeHtml(comp.description || 'No strategic overview provided.')}
            </p>
          </div>

          <div>
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
              <div class="form-label">Configured Sources (${sources.length})</div>
              <a href="#/sources" class="btn btn-ghost btn-sm" onclick="window.state.closeDrawer()">+ Add Source</a>
            </div>

            <div style="display:flex; flex-direction:column; gap:8px;">
              ${sources.length === 0 ? `
                <div style="color:var(--text-muted); font-size:12px; padding:12px; background:var(--bg-canvas); border-radius:6px;">
                  No monitored sources registered for this competitor.
                </div>
              ` : sources.map(s => `
                <div style="background:var(--bg-canvas); border:1px solid var(--border-subtle); padding:10px 14px; border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
                  <div>
                    <div style="font-weight:600; font-size:13px; color:var(--text-primary);">${state.escapeHtml(s.display_name)}</div>
                    <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">${state.escapeHtml(s.url_or_identifier)}</div>
                  </div>
                  <span class="badge ${s.last_fetch_status === 'success' ? 'badge-success' : s.last_fetch_status === 'failure' ? 'badge-danger' : 'badge-neutral'}">
                    ${state.escapeHtml(s.last_fetch_status)}
                  </span>
                </div>
              `).join('')}
            </div>
          </div>

          <div>
            <div class="form-label" style="margin-bottom:10px;">Recent Intelligence Events (${events.length})</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
              ${events.length === 0 ? `
                <div style="color:var(--text-muted); font-size:12px; padding:12px; background:var(--bg-canvas); border-radius:6px;">
                  No intelligence events recorded for this competitor yet.
                </div>
              ` : events.slice(0, 5).map(ev => `
                <div style="background:var(--bg-canvas); border:1px solid var(--border-subtle); padding:12px; border-radius:8px;">
                  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
                    <span class="badge badge-cyan" style="font-size:10px;">${state.escapeHtml(ev.event_type)}</span>
                    <span style="font-size:11px; color:var(--text-muted);">${new Date(ev.detected_at).toLocaleDateString()}</span>
                  </div>
                  <div style="font-weight:600; font-size:13px; color:var(--text-primary);">${state.escapeHtml(ev.title)}</div>
                  <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">${state.escapeHtml(ev.summary || '')}</div>
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
