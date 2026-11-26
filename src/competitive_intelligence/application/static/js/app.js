/**
 * VANTAGE // Application Root Initializer
 * Wires state subscriptions, global layout controls, health probes, and route dispatching.
 */

import { api } from './api.js';
import { state } from './state.js';
import { Router } from './router.js';
import { renderDashboard } from './views/dashboard.js';
import { renderCompetitors } from './views/competitors.js';
import { renderSources } from './views/sources.js';
import { renderIntelligence } from './views/intelligence.js';
import { renderStrategicInsights } from './views/insights.js';
import { renderAsk } from './views/ask.js';
import { renderDecisions } from './views/decisions.js';
import { renderWorkflows } from './views/workflows.js';

export async function initApp() {
  const router = new Router();

  // Register views
  router.register('/overview', renderDashboard);
  router.register('/dashboard', renderDashboard);
  router.register('/competitors', renderCompetitors);
  router.register('/sources', renderSources);
  router.register('/intelligence', renderIntelligence);
  router.register('/insights', renderStrategicInsights);
  router.register('/ask', renderAsk);
  router.register('/decisions', renderStrategicInsights);
  router.register('/workflows', renderWorkflows);

  // Setup Global Workspace Selector
  const wsBtn = document.getElementById('btn-workspace-select');
  const wsNameEl = document.getElementById('active-workspace-name');

  const updateWorkspaceUI = (ws) => {
    if (wsNameEl) {
      wsNameEl.textContent = ws ? ws.name : 'Select Workspace';
    }
  };

  updateWorkspaceUI(state.currentWorkspace);

  state.subscribe((event, data) => {
    if (event === 'workspace_changed') {
      updateWorkspaceUI(data);
      router.handleRoute();
    }
  });

  wsBtn?.addEventListener('click', () => {
    openWorkspaceSwitcherModal(router);
  });

  // Drawer Close Button & Backdrop
  document.getElementById('btn-close-drawer')?.addEventListener('click', () => {
    state.closeDrawer();
  });
  document.getElementById('drawer-backdrop')?.addEventListener('click', () => {
    state.closeDrawer();
  });

  // Modal Close Button
  document.getElementById('btn-close-modal')?.addEventListener('click', () => {
    state.closeModal();
  });

  // Escape key closes modals and drawers
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.closeDrawer();
      state.closeModal();
    }
  });

  // Initial Data Bootstrap: Health & Workspaces
  try {
    const [health, workspaces] = await Promise.all([
      api.getHealth().catch(err => ({ status: 'degraded', error: err.message })),
      api.getWorkspaces().catch(() => []),
    ]);

    state.setHealth(health);
    updateHealthIndicator(health);

    state.setWorkspaces(workspaces);

    // If no workspaces exist at all, auto-prompt creation
    if (workspaces.length === 0) {
      openWorkspaceSwitcherModal(router, true);
    }
  } catch (err) {
    console.error('[INIT ERROR]', err);
  }

  // Periodic Health Probe
  setInterval(async () => {
    try {
      const h = await api.getHealth();
      state.setHealth(h);
      updateHealthIndicator(h);
    } catch (_) {
      updateHealthIndicator({ status: 'offline' });
    }
  }, 30000);

  // Trigger initial route
  await router.handleRoute();
}

function updateHealthIndicator(health) {
  const dot = document.getElementById('health-dot');
  const text = document.getElementById('health-text');
  if (!dot || !text) return;

  dot.className = `status-dot ${health.status === 'ok' ? '' : health.status === 'degraded' ? 'degraded' : 'offline'}`;
  text.textContent = health.status === 'ok' ? 'Systems Nominal' : health.status === 'degraded' ? 'Degraded' : 'Offline';
}

function openWorkspaceSwitcherModal(router, isInitial = false) {
  state.openModal({
    title: 'Switch or Create Workspace',
    renderBody: (el) => {
      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:18px;">
          <div>
            <div class="form-label" style="margin-bottom:8px;">Available Workspaces (${state.workspaces.length})</div>
            <div style="display:flex; flex-direction:column; gap:8px; max-height:200px; overflow-y:auto;" id="ws-modal-list">
              ${state.workspaces.length === 0 ? `
                <div style="color:var(--text-muted); font-size:12px; padding:12px; background:var(--bg-canvas); border-radius:6px;">
                  No workspaces found. Create one below to begin.
                </div>
              ` : state.workspaces.map(ws => `
                <div class="feed-item" data-ws-id="${ws.workspace_id}" style="padding:10px 14px; display:flex; align-items:center; justify-content:space-between; ${state.currentWorkspace?.workspace_id === ws.workspace_id ? 'border-color:var(--accent-primary);' : ''}">
                  <div>
                    <div style="font-weight:600; font-size:13px; color:var(--text-primary);">${state.escapeHtml(ws.name)}</div>
                    <div class="mono-cell" style="font-size:11px;">${state.escapeHtml(ws.workspace_id)}</div>
                  </div>
                  ${state.currentWorkspace?.workspace_id === ws.workspace_id ? '<span class="badge badge-success">ACTIVE</span>' : '<button class="btn btn-secondary btn-sm">Select</button>'}
                </div>
              `).join('')}
            </div>
          </div>

          <div style="border-top:1px solid var(--border-subtle); padding-top:16px;">
            <div class="form-label" style="margin-bottom:10px;">Create New Workspace</div>
            <div style="display:flex; gap:10px;">
              <input type="text" class="form-control" id="new-ws-name" placeholder="e.g. Enterprise SaaS Monitor" />
              <button class="btn btn-primary" id="btn-create-ws">Create</button>
            </div>
          </div>
        </div>
      `;

      // Wire select existing
      el.querySelectorAll('#ws-modal-list .feed-item').forEach(item => {
        item.addEventListener('click', () => {
          const wsId = item.dataset.wsId;
          const found = state.workspaces.find(w => w.workspace_id === wsId);
          if (found) {
            state.setWorkspace(found);
            state.showToast(`Switched to workspace: ${found.name}`, 'info');
            state.closeModal();
          }
        });
      });

      // Wire create new
      el.querySelector('#btn-create-ws')?.addEventListener('click', async () => {
        const nameInput = el.querySelector('#new-ws-name');
        const name = nameInput?.value.trim();
        if (!name) {
          state.showToast('Workspace name cannot be empty', 'warning');
          return;
        }

        const createBtn = el.querySelector('#btn-create-ws');
        if (createBtn) createBtn.disabled = true;

        try {
          const newWs = await api.createWorkspace({
            name,
            owner_user_id: state.currentUserId,
          });
          const all = await api.getWorkspaces();
          state.setWorkspaces(all);
          state.setWorkspace(newWs);
          state.showToast(`Workspace '${name}' created`, 'success');
          state.closeModal();
        } catch (err) {
          state.showToast(err.message, 'error');
          if (createBtn) createBtn.disabled = false;
        }
      });
    },
    renderFooter: () => isInitial ? '' : `
      <button class="btn btn-secondary btn-sm" onclick="window.state.closeModal()">Close</button>
    `,
  });
}

// Auto-run on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});
