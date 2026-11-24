/**
 * VANTAGE // Sources & Monitoring View
 * Monitored source perimeter configuration, fetch telemetry, and on-demand ingestion dispatch.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderSources(container) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
      <div>
        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Monitored Sources</h2>
        <p style="color:var(--text-muted); font-size:13px;">Manage observation endpoints, polling schedules, and automated crawler runs</p>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <select class="form-control" id="filter-competitor" style="width:180px; padding:6px 12px; font-size:12px;">
          <option value="">All Competitors</option>
        </select>
        <button class="btn btn-primary" id="btn-add-source">
          <span>+</span> Register Source
        </button>
      </div>
    </div>

    <div id="monitoring-diagnostics-container" style="margin-top:14px;"></div>

    <div class="table-container" id="sources-table-container">
      <div style="padding:40px; text-align:center;" class="skeleton">Loading monitored sources...</div>
    </div>
  `;

  try {
    const [competitors, sources, diagnostics] = await Promise.all([
      api.getCompetitors(wsId),
      api.getSources(wsId),
      api.getMonitoringDiagnostics(wsId).catch(() => null),
    ]);

    // Render Diagnostics Summary if available
    const diagContainer = document.getElementById('monitoring-diagnostics-container');
    if (diagContainer && diagnostics) {
      diagContainer.innerHTML = `
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin-bottom:16px;">
          <div class="stat-card" style="padding:12px 14px;">
            <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Monitored Perimeter</div>
            <div style="font-size:18px; font-weight:700; margin-top:2px;">${diagnostics.total_sources} Sources</div>
            <div style="font-size:10px; color:var(--text-muted);">${diagnostics.disabled_sources_count} Polling Disabled</div>
          </div>
          <div class="stat-card" style="padding:12px 14px;">
            <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Acquisition Health</div>
            <div style="font-size:18px; font-weight:700; color:var(--status-success); margin-top:2px;">${diagnostics.healthy_sources_count} Healthy</div>
            <div style="font-size:10px; color:${diagnostics.failing_sources_count > 0 ? 'var(--status-danger)' : 'var(--text-muted)'};">
              ${diagnostics.failing_sources_count} Failing Sources
            </div>
          </div>
          <div class="stat-card" style="padding:12px 14px;">
            <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Active Leases</div>
            <div style="font-size:18px; font-weight:700; color:var(--accent-cyan); margin-top:2px;">${diagnostics.active_leases_count} In-Flight</div>
            <div style="font-size:10px; color:var(--text-muted);">Concurrency Lock Guarded</div>
          </div>
          <div class="stat-card" style="padding:12px 14px;">
            <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Scheduler Engine</div>
            <div style="font-size:14px; font-weight:600; color:var(--text-primary); margin-top:4px;">${state.escapeHtml(diagnostics.scheduler_mode.toUpperCase())}</div>
            <div style="font-size:10px; color:${diagnostics.scheduler_running ? 'var(--status-success)' : 'var(--status-danger)'};">
              ${diagnostics.scheduler_running ? '● Polling Active' : '○ Disabled'}
            </div>
          </div>
        </div>
      `;
    }

    const compMap = new Map();
    competitors.forEach(c => compMap.set(c.competitor_id, c.name));


    // Populate filter dropdown
    const filterSelect = document.getElementById('filter-competitor');
    if (filterSelect) {
      competitors.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.competitor_id;
        opt.textContent = c.name;
        filterSelect.appendChild(opt);
      });

      filterSelect.addEventListener('change', () => {
        renderTable(filterSelect.value);
      });
    }

    document.getElementById('btn-add-source')?.addEventListener('click', () => {
      openAddSourceModal(wsId, competitors, () => renderSources(container));
    });

    const renderTable = (filterCompId = null) => {
      const tableContainer = document.getElementById('sources-table-container');
      if (!tableContainer) return;

      const filtered = filterCompId
        ? sources.filter(s => s.company_id === filterCompId)
        : sources;

      if (filtered.length === 0) {
        tableContainer.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">📡</div>
            <h3 class="empty-title">No Monitored Sources Found</h3>
            <p class="empty-description">
              ${filterCompId ? 'No sources configured for this competitor.' : 'Register competitor web endpoints, pricing matrices, release notes, or job boards to monitor.'}
            </p>
            <button class="btn btn-primary btn-sm" id="btn-empty-add-src">Register Source</button>
          </div>
        `;
        tableContainer.querySelector('#btn-empty-add-src')?.addEventListener('click', () => {
          openAddSourceModal(wsId, competitors, () => renderSources(container));
        });
        return;
      }

      tableContainer.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Source / Display Name</th>
              <th>Competitor</th>
              <th>Category</th>
              <th>Schedule / Priority</th>
              <th>Polling</th>
              <th>Acquisition Health</th>
              <th style="text-align:right;">Actions</th>
            </tr>
          </thead>
          <tbody>
            ${filtered.map(src => `
              <tr data-source-id="${src.source_id}">
                <td>
                  <div style="font-weight:600; color:var(--text-primary);">${state.escapeHtml(src.display_name)}</div>
                  <div class="mono-cell" style="max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    ${state.escapeHtml(src.url_or_identifier)}
                  </div>
                </td>
                <td>
                  <span style="font-weight:500; color:var(--accent-cyan);">${state.escapeHtml(compMap.get(src.company_id) || src.company_id)}</span>
                </td>
                <td>
                  <span class="badge badge-purple">${state.escapeHtml(src.category)}</span>
                </td>
                <td>
                  <div style="font-size:12px;">${formatFrequency(src.frequency_seconds)}</div>
                  <div class="mono-cell" style="font-size:10px;">Priority ${src.priority}/10</div>
                </td>
                <td>
                  <span class="badge ${src.monitoring_enabled ? 'badge-success' : 'badge-neutral'}">
                    ${src.monitoring_enabled ? 'ENABLED' : 'DISABLED'}
                  </span>
                </td>
                <td>
                  <div style="display:flex; flex-direction:column; gap:4px;">
                    <div style="display:flex; align-items:center; gap:6px;">
                      <span class="badge ${src.last_fetch_status === 'success' ? 'badge-success' : src.last_fetch_status === 'failure' ? 'badge-danger' : 'badge-warning'}">
                        ${state.escapeHtml(src.last_fetch_status)}
                      </span>
                      ${src.consecutive_failure_count > 0 ? `
                        <span class="badge badge-danger" style="font-size:10px; padding:2px 5px;" title="${src.consecutive_failure_count} consecutive monitoring failures">
                          ⚠ ${src.consecutive_failure_count} fails
                        </span>
                      ` : ''}
                    </div>
                    ${src.last_checked_at ? `
                      <div class="mono-cell" style="font-size:10px; color:var(--text-muted);">
                        Checked: ${new Date(src.last_checked_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    ` : ''}
                  </div>
                </td>

                <td style="text-align:right;">
                  <button class="btn btn-primary btn-sm btn-trigger-source" data-source-id="${src.source_id}">
                    Trigger Run
                  </button>
                  <button class="btn btn-ghost btn-sm btn-edit-source" data-source-id="${src.source_id}">
                    Policy
                  </button>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;

      // Wire action buttons
      tableContainer.querySelectorAll('.btn-trigger-source').forEach(btn => {
        btn.addEventListener('click', async () => {
          const sId = btn.dataset.sourceId;
          btn.disabled = true;
          btn.textContent = 'Triggering...';
          try {
            const run = await api.triggerSource(wsId, sId, { target_stop_stage: 'insight_generation' });
            state.showToast(`Ingestion workflow dispatched: ${run.run_id.substring(0, 8)}...`, 'success');
            // Navigate to workflows to see live progress
            window.location.hash = `#/workflows`;
          } catch (err) {
            state.showToast(err.message, 'error');
            btn.disabled = false;
            btn.textContent = 'Trigger Run';
          }
        });
      });

      tableContainer.querySelectorAll('.btn-edit-source').forEach(btn => {
        btn.addEventListener('click', () => {
          const sId = btn.dataset.sourceId;
          const src = sources.find(s => s.source_id === sId);
          if (src) openEditSourceModal(wsId, src, () => renderSources(container));
        });
      });
    };

    renderTable();

  } catch (err) {
    container.innerHTML = `
      <div class="card" style="border-color:var(--status-danger);">
        <h3 style="color:var(--status-danger);">Failed Loading Sources</h3>
        <p style="color:var(--text-secondary);">${state.escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function formatFrequency(seconds) {
  if (!seconds) return 'On-demand';
  if (seconds < 3600) return `Every ${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `Every ${Math.round(seconds / 3600)}h`;
  return `Every ${Math.round(seconds / 86400)}d`;
}

function openAddSourceModal(workspaceId, competitors, onSuccess) {
  state.openModal({
    title: 'Register Monitored Source',
    renderBody: (el) => {
      el.innerHTML = `
        <form id="form-register-source" style="display:flex; flex-direction:column; gap:16px;">
          <div class="form-group">
            <label class="form-label">Competitor *</label>
            <select class="form-control" id="src-comp" required>
              ${competitors.map(c => `<option value="${c.competitor_id}">${state.escapeHtml(c.name)}</option>`).join('')}
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Display Name *</label>
            <input type="text" class="form-control" id="src-name" placeholder="e.g. Official Pricing Matrix" required />
          </div>
          <div class="form-group">
            <label class="form-label">URL or Endpoint Identifier *</label>
            <input type="url" class="form-control" id="src-url" placeholder="https://example.com/pricing" required />
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div class="form-group">
              <label class="form-label">Intelligence Category</label>
              <select class="form-control" id="src-cat">
                <option value="pricing">Pricing & Tiers</option>
                <option value="product">Product & Features</option>
                <option value="hiring">Hiring & Talent</option>
                <option value="news">Corporate News</option>
                <option value="social">Social & Platform</option>
                <option value="reviews">Customer Reviews</option>
                <option value="financial">Financial & Earnings</option>
                <option value="legal_regulatory">Legal & Regulatory</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">Polling Frequency</label>
              <select class="form-control" id="src-freq">
                <option value="3600">Hourly (3,600s)</option>
                <option value="21600">Every 6 Hours</option>
                <option value="86400" selected>Daily (86,400s)</option>
                <option value="604800">Weekly</option>
              </select>
            </div>
          </div>
        </form>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-ghost" onclick="window.state.closeModal()">Cancel</button>
      <button class="btn btn-primary" id="btn-submit-source">Register Source</button>
    `,
  });

  document.getElementById('btn-submit-source')?.addEventListener('click', async () => {
    const companyId = document.getElementById('src-comp')?.value;
    const displayName = document.getElementById('src-name')?.value.trim();
    const url = document.getElementById('src-url')?.value.trim();
    const category = document.getElementById('src-cat')?.value;
    const frequency = parseInt(document.getElementById('src-freq')?.value || '86400', 10);

    if (!companyId || !displayName || !url) {
      state.showToast('Please fill all required fields', 'warning');
      return;
    }

    const btn = document.getElementById('btn-submit-source');
    if (btn) btn.disabled = true;

    try {
      await api.createSource(workspaceId, {
        company_id: companyId,
        display_name: displayName,
        url_or_identifier: url,
        category: category,
        frequency_seconds: frequency,
        priority: 1,
      });
      state.showToast(`Source '${displayName}' registered`, 'success');
      state.closeModal();
      if (onSuccess) onSuccess();
    } catch (err) {
      state.showToast(err.message, 'error');
      if (btn) btn.disabled = false;
    }
  });
}

function openEditSourceModal(workspaceId, source, onSuccess) {
  state.openModal({
    title: `Monitoring Policy: ${source.display_name}`,
    renderBody: (el) => {
      el.innerHTML = `
        <form style="display:flex; flex-direction:column; gap:16px;">
          <div class="form-group">
            <label class="form-label">Monitoring Active</label>
            <select class="form-control" id="edit-enabled">
              <option value="true" ${source.monitoring_enabled ? 'selected' : ''}>Active Polling Enabled</option>
              <option value="false" ${!source.monitoring_enabled ? 'selected' : ''}>Paused / Disabled</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Polling Frequency (seconds)</label>
            <input type="number" class="form-control" id="edit-freq" value="${source.frequency_seconds}" min="60" />
            <div class="form-help">Minimum frequency is 60 seconds</div>
          </div>
          <div class="form-group">
            <label class="form-label">Priority (1 to 10)</label>
            <input type="number" class="form-control" id="edit-priority" value="${source.priority}" min="1" max="10" />
          </div>
        </form>
      `;
    },
    renderFooter: () => `
      <button class="btn btn-ghost" onclick="window.state.closeModal()">Cancel</button>
      <button class="btn btn-primary" id="btn-save-policy">Save Changes</button>
    `,
  });

  document.getElementById('btn-save-policy')?.addEventListener('click', async () => {
    const enabled = document.getElementById('edit-enabled')?.value === 'true';
    const frequency = parseInt(document.getElementById('edit-freq')?.value || '86400', 10);
    const priority = parseInt(document.getElementById('edit-priority')?.value || '1', 10);

    const btn = document.getElementById('btn-save-policy');
    if (btn) btn.disabled = true;

    try {
      await api.updateSource(workspaceId, source.source_id, {
        monitoring_enabled: enabled,
        frequency_seconds: frequency,
        priority: priority,
      });
      state.showToast('Monitoring policy updated', 'success');
      state.closeModal();
      if (onSuccess) onSuccess();
    } catch (err) {
      state.showToast(err.message, 'error');
      if (btn) btn.disabled = false;
    }
  });
}
