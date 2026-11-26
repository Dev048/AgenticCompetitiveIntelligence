/**
 * VANTAGE // Competitive Intelligence API Client
 * Robust asynchronous client for Subsystem 6 REST endpoints.
 */

export const API_CONFIG = {
  baseUrl: window.VANTAGE_API_BASE || '/api/v1',
  healthUrl: window.VANTAGE_HEALTH_URL || '/health',
};

export class ApiClient {
  constructor(baseUrl = API_CONFIG.baseUrl) {
    this.baseUrl = baseUrl;
  }

  generateCorrelationId() {
    return 'req_' + Math.random().toString(36).substring(2, 11);
  }

  async request(endpoint, options = {}) {
    const correlationId = this.generateCorrelationId();
    const headers = {
      'Content-Type': 'application/json',
      'X-Correlation-ID': correlationId,
      ...options.headers,
    };

    const url = endpoint.startsWith('http')
      ? endpoint
      : endpoint.startsWith('/health')
        ? API_CONFIG.healthUrl
        : `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        let errorData = null;
        try {
          errorData = await response.json();
        } catch (_) {
          errorData = { detail: await response.text() };
        }
        const error = new Error(
          errorData.detail || errorData.message || `API Error: ${response.status} ${response.statusText}`
        );
        error.status = response.status;
        error.code = errorData.error_code || null;
        error.data = errorData;
        throw error;
      }

      if (response.status === 204) {
        return null;
      }

      return await response.json();
    } catch (err) {
      // Expected client errors (rate limit, read-only demo, validation, not found) are
      // shown to the user by the calling view; only server/network failures are errors.
      const log = err.status && err.status < 500 ? console.warn : console.error;
      log(`[API] ${options.method || 'GET'} ${url}:`, err.message);
      throw err;
    }
  }

  // Health
  getHealth() {
    return this.request('/health');
  }

  // Workspaces
  getWorkspaces(userId = null) {
    const query = userId ? `?user_id=${encodeURIComponent(userId)}` : '';
    return this.request(`/workspaces${query}`);
  }

  createWorkspace(data) {
    return this.request('/workspaces', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  getWorkspace(workspaceId) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}`);
  }

  // Competitors
  getCompetitors(workspaceId, isActive = null) {
    const query = isActive !== null ? `?is_active=${isActive}` : '';
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/competitors${query}`);
  }

  createCompetitor(workspaceId, data) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/competitors`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  getCompetitor(workspaceId, competitorId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/competitors/${encodeURIComponent(competitorId)}`
    );
  }

  updateCompetitor(workspaceId, competitorId, data) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/competitors/${encodeURIComponent(competitorId)}`,
      {
        method: 'PATCH',
        body: JSON.stringify(data),
      }
    );
  }

  // Sources
  getSources(workspaceId, companyId = null, category = null) {
    const params = new URLSearchParams();
    if (companyId) params.append('company_id', companyId);
    if (category) params.append('category', category);
    const qs = params.toString() ? `?${params.toString()}` : '';
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/sources${qs}`);
  }

  createSource(workspaceId, data) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/sources`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  updateSource(workspaceId, sourceId, data) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/sources/${encodeURIComponent(sourceId)}`,
      {
        method: 'PATCH',
        body: JSON.stringify(data),
      }
    );
  }

  triggerSource(workspaceId, sourceId, data = {}) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/sources/${encodeURIComponent(sourceId)}/trigger`,
      {
        method: 'POST',
        body: JSON.stringify(data),
      }
    );
  }

  getMonitoringDiagnostics(workspaceId) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/sources/monitoring/diagnostics`);
  }


  // Intelligence: Events, Signals, Insights
  getEvents(workspaceId, companyId = null, limit = 100) {
    const params = new URLSearchParams({ limit });
    if (companyId) params.append('company_id', companyId);
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/events?${params.toString()}`);
  }

  getEvent(workspaceId, eventId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/events/${encodeURIComponent(eventId)}`
    );
  }

  getSignals(workspaceId, companyId = null, limit = 100) {
    const params = new URLSearchParams({ limit });
    if (companyId) params.append('company_id', companyId);
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/signals?${params.toString()}`);
  }

  getSignal(workspaceId, signalId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/signals/${encodeURIComponent(signalId)}`
    );
  }

  getInsights(workspaceId, companyId = null, limit = 100) {
    const params = new URLSearchParams({ limit });
    if (companyId) params.append('company_id', companyId);
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/insights?${params.toString()}`);
  }

  getInsight(workspaceId, insightId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/insights/${encodeURIComponent(insightId)}`
    );
  }

  // Decisions & Recommendations
  getDecisionContexts(workspaceId) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/decisions/contexts`);
  }

  createDecisionContext(workspaceId, data) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/decisions/contexts`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  getDecisionContext(workspaceId, contextId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/decisions/contexts/${encodeURIComponent(contextId)}`
    );
  }

  getDecisionOptions(workspaceId, contextId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/decisions/contexts/${encodeURIComponent(contextId)}/options`
    );
  }

  getRecommendations(workspaceId, contextId = null) {
    const query = contextId ? `?context_id=${encodeURIComponent(contextId)}` : '';
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/recommendations${query}`);
  }

  getRecommendation(workspaceId, recommendationId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/recommendations/${encodeURIComponent(recommendationId)}`
    );
  }

  // Evidence query (retrieval + bounded agentic RAG trace)
  queryEvidence(workspaceId, payload) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/query`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  getRecommendationProvenance(workspaceId, recommendationId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/recommendations/${encodeURIComponent(recommendationId)}/provenance`
    );
  }

  updateRecommendationStatus(workspaceId, recommendationId, status, reviewNotes = null) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/recommendations/${encodeURIComponent(recommendationId)}/status`,
      {
        method: 'PATCH',
        body: JSON.stringify({ status, review_notes: reviewNotes }),
      }
    );
  }

  // Workflows
  getWorkflowRuns(workspaceId, companyId = null, limit = 50) {
    const params = new URLSearchParams({ limit });
    if (companyId) params.append('company_id', companyId);
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/workflows/runs?${params.toString()}`);
  }

  getWorkflowRun(workspaceId, runId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/workflows/runs/${encodeURIComponent(runId)}`
    );
  }

  submitWorkflowRun(workspaceId, data) {
    return this.request(`/workspaces/${encodeURIComponent(workspaceId)}/workflows/runs`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  resumeWorkflowRun(workspaceId, runId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/workflows/runs/${encodeURIComponent(runId)}/resume`,
      {
        method: 'POST',
      }
    );
  }

  cancelWorkflowRun(workspaceId, runId) {
    return this.request(
      `/workspaces/${encodeURIComponent(workspaceId)}/workflows/runs/${encodeURIComponent(runId)}/cancel`,
      {
        method: 'POST',
      }
    );
  }
}

export const api = new ApiClient();
