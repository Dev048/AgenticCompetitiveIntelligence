/**
 * VANTAGE // Client Router
 * Hash-based client router orchestrating views, history, and active navigation states.
 */

import { state } from './state.js';

export class Router {
  constructor(routes = {}) {
    this.routes = routes;
    this.currentPath = '';
    this.params = {};

    window.addEventListener('hashchange', () => this.handleRoute());
  }

  register(path, handler) {
    this.routes[path] = handler;
  }

  navigate(path) {
    window.location.hash = path.startsWith('#') ? path : `#${path}`;
  }

  parseHash() {
    const rawHash = window.location.hash.slice(1) || '/dashboard';
    const [pathPart, queryPart] = rawHash.split('?');
    const path = pathPart.startsWith('/') ? pathPart : `/${pathPart}`;

    const query = {};
    if (queryPart) {
      const sp = new URLSearchParams(queryPart);
      for (const [k, v] of sp.entries()) {
        query[k] = v;
      }
    }

    return { path, query };
  }

  async handleRoute() {
    const { path, query } = this.parseHash();
    this.currentPath = path;

    // Update active nav links
    const navLinks = document.querySelectorAll('.nav-item');
    navLinks.forEach(link => {
      const href = link.getAttribute('href') || '';
      const linkPath = href.replace('#', '').split('?')[0];
      if (linkPath === path || (path === '/' && linkPath === '/dashboard')) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Update breadcrumbs
    const breadcrumbRoot = document.getElementById('breadcrumb-root');
    const breadcrumbCurrent = document.getElementById('breadcrumb-current');
    if (breadcrumbRoot) {
      breadcrumbRoot.textContent = state.currentWorkspace ? state.currentWorkspace.name : 'Workspace';
    }
    if (breadcrumbCurrent) {
      const title = path.replace('/', '').replace(/_/g, ' ');
      breadcrumbCurrent.textContent = title ? title.charAt(0).toUpperCase() + title.slice(1) : 'Dashboard';
    }

    // Match route handler
    const handler = this.routes[path] || this.routes['*'] || this.routes['/dashboard'];
    const viewport = document.getElementById('content-viewport');
    if (!viewport) return;

    if (!state.currentWorkspace && path !== '/workspaces') {
      // If no workspace is selected and none exist, render onboarding
      this.renderNoWorkspaceState(viewport);
      return;
    }

    try {
      if (handler) {
        await handler(viewport, { path, query });
      }
    } catch (err) {
      console.error(`[ROUTER ERROR] Failed navigating to ${path}:`, err);
      viewport.innerHTML = `
        <div class="card" style="border-color: var(--status-danger);">
          <div class="card-header">
            <h3 class="card-title" style="color: var(--status-danger);">Navigation Error</h3>
          </div>
          <p style="color: var(--text-secondary);">An error occurred while loading this view: ${state.escapeHtml(err.message)}</p>
          <div style="margin-top: 12px;">
            <button class="btn btn-secondary" onclick="window.location.reload()">Reload Page</button>
          </div>
        </div>
      `;
    }
  }

  renderNoWorkspaceState(container) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🏢</div>
        <h3 class="empty-title">No Workspace Selected</h3>
        <p class="empty-description">
          Competitive Intelligence is scoped by workspace. Create or select a workspace to monitor competitors, analyze signals, and formulate strategic decisions.
        </p>
        <button class="btn btn-primary" id="btn-create-initial-ws">Create Workspace</button>
      </div>
    `;

    container.querySelector('#btn-create-initial-ws')?.addEventListener('click', () => {
      document.getElementById('btn-workspace-select')?.click();
    });
  }
}
