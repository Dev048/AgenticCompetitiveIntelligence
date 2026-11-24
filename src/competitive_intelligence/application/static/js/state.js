/**
 * VANTAGE // Application State Store
 * Manages active workspace, global notifications, drawers, and cached entities.
 */

class StateStore {
  constructor() {
    this.currentWorkspace = null;
    this.workspaces = [];
    this.currentUserId = 'default_user';
    this.health = { status: 'unknown', environment: 'unknown' };
    this.listeners = new Set();
    this.activeDrawer = null;
    this.activeModal = null;

    // Load persisted workspace from localStorage if available
    const savedWs = localStorage.getItem('vantage_active_workspace');
    if (savedWs) {
      try {
        this.currentWorkspace = JSON.parse(savedWs);
      } catch (_) {
        this.currentWorkspace = null;
      }
    }
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(event, payload) {
    for (const listener of this.listeners) {
      try {
        listener(event, payload);
      } catch (err) {
        console.error('[STATE NOTIFY ERROR]', err);
      }
    }
  }

  setWorkspaces(workspaces) {
    this.workspaces = workspaces || [];
    if (!this.currentWorkspace && this.workspaces.length > 0) {
      this.setWorkspace(this.workspaces[0]);
    } else if (this.currentWorkspace) {
      const match = this.workspaces.find(w => w.workspace_id === this.currentWorkspace.workspace_id);
      if (match) {
        this.currentWorkspace = match;
      }
    }
    this.notify('workspaces_updated', this.workspaces);
  }

  setWorkspace(workspace) {
    this.currentWorkspace = workspace;
    if (workspace) {
      localStorage.setItem('vantage_active_workspace', JSON.stringify(workspace));
    } else {
      localStorage.removeItem('vantage_active_workspace');
    }
    this.notify('workspace_changed', this.currentWorkspace);
  }

  setHealth(health) {
    this.health = health;
    this.notify('health_updated', this.health);
  }

  // Toast System
  showToast(message, type = 'info', durationMs = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div style="flex:1;">${this.escapeHtml(message)}</div>
      <button class="btn btn-ghost btn-sm btn-icon" style="padding:2px 6px;">&times;</button>
    `;

    const closeBtn = toast.querySelector('button');
    const dismiss = () => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 200);
    };

    closeBtn.onclick = dismiss;
    container.appendChild(toast);

    if (durationMs > 0) {
      setTimeout(dismiss, durationMs);
    }
  }

  // Slide-Over Drawer
  openDrawer({ title, subtitle, renderContent, renderFooter }) {
    this.activeDrawer = { title, subtitle, renderContent, renderFooter };
    const drawer = document.getElementById('slide-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    const titleEl = document.getElementById('drawer-title');
    const subtitleEl = document.getElementById('drawer-subtitle');
    const bodyEl = document.getElementById('drawer-body');
    const footerEl = document.getElementById('drawer-footer');

    if (titleEl) titleEl.textContent = title || '';
    if (subtitleEl) subtitleEl.textContent = subtitle || '';
    if (bodyEl) {
      bodyEl.innerHTML = '';
      if (typeof renderContent === 'function') {
        const content = renderContent(bodyEl);
        if (content && typeof content === 'string') {
          bodyEl.innerHTML = content;
        }
      }
    }
    if (footerEl) {
      footerEl.innerHTML = '';
      if (typeof renderFooter === 'function') {
        const footer = renderFooter(footerEl);
        if (footer && typeof footer === 'string') {
          footerEl.innerHTML = footer;
        }
        footerEl.style.display = 'flex';
      } else {
        footerEl.style.display = 'none';
      }
    }

    drawer?.classList.add('open');
    backdrop?.classList.add('open');
  }

  closeDrawer() {
    this.activeDrawer = null;
    const drawer = document.getElementById('slide-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    drawer?.classList.remove('open');
    backdrop?.classList.remove('open');
  }

  // Modal System
  openModal({ title, renderBody, renderFooter }) {
    this.activeModal = { title, renderBody, renderFooter };
    const overlay = document.getElementById('modal-overlay');
    const titleEl = document.getElementById('modal-title');
    const bodyEl = document.getElementById('modal-body');
    const footerEl = document.getElementById('modal-footer');

    if (titleEl) titleEl.textContent = title || '';
    if (bodyEl) {
      bodyEl.innerHTML = '';
      if (typeof renderBody === 'function') {
        const content = renderBody(bodyEl);
        if (content && typeof content === 'string') {
          bodyEl.innerHTML = content;
        }
      }
    }
    if (footerEl) {
      footerEl.innerHTML = '';
      if (typeof renderFooter === 'function') {
        const footer = renderFooter(footerEl);
        if (footer && typeof footer === 'string') {
          footerEl.innerHTML = footer;
        }
        footerEl.style.display = 'flex';
      } else {
        footerEl.style.display = 'none';
      }
    }

    overlay?.classList.add('open');
  }

  closeModal() {
    this.activeModal = null;
    const overlay = document.getElementById('modal-overlay');
    overlay?.classList.remove('open');
  }

  escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

export const state = new StateStore();
