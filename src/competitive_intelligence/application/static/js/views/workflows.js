/**
 * VANTAGE // Workflows View
 * Operational pipeline executions, stage telemetry, stage audit trail, and resume/cancel actions.
 */

import { api } from '../api.js';
import { state } from '../state.js';

export async function renderWorkflows(container) {
  const wsId = state.currentWorkspace?.workspace_id;
  if (!wsId) return;

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
      <div>
        <h2 style="font-family:var(--font-display); font-size:20px; font-weight:700;">Workflow Orchestration</h2>
        <p style="color:var(--text-muted); font-size:13px;">
          Deterministic stage machine: Knowledge Ingestion &rarr; Retrieval &rarr; Event Analysis &rarr; Signal Detection &rarr; Insight Generation &rarr; Decision Support
        </p>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <button class="btn btn-secondary btn-sm" id="btn-refresh-workflows">
          <span>↻</span> Refresh Runs
        </button>
        <button class="btn btn-primary" id="btn-dispatch-workflow">
          <span>+</span> Execute Operational Workflow
        </button>
      </div>
    </div>

    <div class="table-container" id="workflows-table-container">
      <div style="padding:40px; text-align:center;" class="skeleton">Loading workflow execution telemetry...</div>
    </div>
  `;

  document.getElementById('btn-refresh-workflows')?.addEventListener('click', () => {
    renderWorkflows(container);
  });

  try {
    const [workflows, competitors, contexts] = await Promise.all([
      api.getWorkflowRuns(wsId, null, 50),
      api.getCompetitors(wsId).catch(() => []),
      api.getDecisionContexts(wsId).catch(() => []),
    ]);

    const compMap = new Map();
    competitors.forEach(c => compMap.set(c.competitor_id, c.name));

    document.getElementById('btn-dispatch-workflow')?.addEventListener('click', () => {
      openDispatchWorkflowModal(wsId, competitors, contexts, () => renderWorkflows(container));
    });

    const tableContainer = document.getElementById('workflows-table-container');
    if (!tableContainer) return;

    if (workflows.length === 0) {
      tableContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚙️</div>
          <h3 class="empty-title">No Workflow Executions Recorded</h3>
          <p class="empty-description">
            Operational workflows execute deterministic normalization, embedding, retrieval, LLM event analysis, and decision support with checkpoint state persistence.
          </p>
          <button class="btn btn-primary" id="btn-empty-dispatch">Dispatch First Workflow</button>
        </div>
      `;
      tableContainer.querySelector('#btn-empty-dispatch')?.addEventListener('click', () => {
        openDispatchWorkflowModal(wsId, competitors, contexts, () => renderWorkflows(container));
      });
      return;
    }

    tableContainer.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Run ID / Trigger</th>
            <th>Competitor</th>
            <th>Status</th>
            <th>Pipeline Stage Progress</th>
            <th>Duration</th>
            <th>Started</th>
            <th style="text-align:right;">Actions</th>
          </tr>
        </thead>
        <tbody>
          ${workflows.map(run => {
            const isFinished = run.status === 'completed' || run.status === 'failed' || run.status === 'cancelled';
            const durationSec = run.completed_at && run.started_at
              ? ((new Date(run.completed_at) - new Date(run.started_at)) / 1000).toFixed(1) + 's'
              : 'in-flight';

            return `
              <tr data-run-id="${run.run_id}">
                <td>
                  <div style="font-family:var(--font-mono); font-weight:600; font-size:12px; color:var(--text-primary);">
                    ${state.escapeHtml(run.run_id.substring(0, 16))}...
                  </div>
                  <div style="font-size:11px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                    <span>Trigger: <span style="color:var(--text-secondary);">${state.escapeHtml(run.trigger_type)}</span></span>
                    ${run.correlation_id ? `
                      <span>&bull;</span>
                      <button class="btn-copy-corr" data-corr="${state.escapeHtml(run.correlation_id)}" style="background:none; border:none; padding:0; cursor:pointer; font-size:10px; color:var(--accent-cyan);" title="Copy Correlation ID: ${state.escapeHtml(run.correlation_id)}">
                        📋 ${state.escapeHtml(run.correlation_id.substring(0, 8))}
                      </button>
                    ` : ''}
                  </div>
                </td>
                <td>
                  <span style="font-weight:600; color:var(--accent-cyan); font-size:12px;">
                    ${state.escapeHtml(compMap.get(run.company_id) || run.company_id || 'Global')}
                  </span>
                </td>
                <td>
                  <span class="badge ${run.status === 'completed' ? 'badge-success' : run.status === 'failed' ? 'badge-danger' : run.status === 'running' ? 'badge-purple' : 'badge-neutral'}">
                    ${state.escapeHtml(run.status.toUpperCase())}
                  </span>
                </td>
                <td>
                  ${renderPipelineBadges(run)}
                </td>
                <td class="mono-cell" style="font-size:12px;">
                  ${durationSec}
                </td>
                <td class="mono-cell" style="font-size:11px;">
                  ${new Date(run.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </td>
                <td style="text-align:right;">
                  <button class="btn btn-secondary btn-sm btn-inspect-run" data-run-id="${run.run_id}">
                    Audit
                  </button>
                  ${run.status === 'failed' ? `
                    <button class="btn btn-primary btn-sm btn-resume-run" data-run-id="${run.run_id}">
                      Resume
                    </button>
                  ` : ''}
                  ${run.status === 'running' || run.status === 'resuming' ? `
                    <button class="btn btn-danger btn-sm btn-cancel-run" data-run-id="${run.run_id}">
                      Cancel
                    </button>
                  ` : ''}
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    `;

    // Wire Correlation ID copy buttons
    tableContainer.querySelectorAll('.btn-copy-corr').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const text = btn.dataset.corr;
        if (text) {
          navigator.clipboard.writeText(text);
          state.showToast('Correlation ID copied to clipboard', 'info');
        }
      });
    });


    // Wire actions
    tableContainer.querySelectorAll('.btn-inspect-run').forEach(btn => {
      btn.addEventListener('click', () => {
        const rId = btn.dataset.runId;
        const run = workflows.find(w => w.run_id === rId);
        if (run) openWorkflowAuditDrawer(run);
      });
    });

    tableContainer.querySelectorAll('.btn-resume-run').forEach(btn => {
      btn.addEventListener('click', async () => {
        const rId = btn.dataset.runId;
        btn.disabled = true;
        btn.textContent = 'Resuming...';
        try {
          await api.resumeWorkflowRun(wsId, rId);
          state.showToast('Workflow resumed from checkpoint', 'success');
          renderWorkflows(container);
        } catch (err) {
          state.showToast(err.message, 'error');
          btn.disabled = false;
          btn.textContent = 'Resume';
        }
      });
    });

    tableContainer.querySelectorAll('.btn-cancel-run').forEach(btn => {
      btn.addEventListener('click', async () => {
        const rId = btn.dataset.runId;
        btn.disabled = true;
        try {
          await api.cancelWorkflowRun(wsId, rId);
          state.showToast('Workflow cancelled', 'info');
          renderWorkflows(container);
        } catch (err) {
          state.showToast(err.message, 'error');
          btn.disabled = false;
        }
      });
    });

  } catch (err) {
    container.innerHTML = `
      <div class="card" style="border-color:var(--status-danger);">
        <h3 style="color:var(--status-danger);">Failed Loading Workflows</h3>
        <p style="color:var(--text-secondary);">${state.escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function renderPipelineBadges(run) {
  const stages = [
    { key: 'knowledge_processing', label: 'KP' },
    { key: 'retrieval', label: 'RET' },
    { key: 'event_analysis', label: 'EA' },
    { key: 'signal_detection', label: 'SD' },
    { key: 'insight_generation', label: 'IG' },
    { key: 'decision_support', label: 'DS' },
  ];

  const targetIndex = stages.findIndex(s => s.key === run.target_stop_stage);
  const targetStages = targetIndex >= 0 ? stages.slice(0, targetIndex + 1) : stages;

  const records = run.stage_records || {};

  return `
    <div class="workflow-pipeline">
      ${targetStages.map(s => {
        const rec = records[s.key];
        const status = rec?.status || (run.status === 'failed' ? 'skipped' : 'pending');
        const badgeClass = status === 'completed'
          ? 'completed'
          : status === 'running'
            ? 'running'
            : status === 'failed'
              ? 'failed'
              : '';

        return `
          <span class="pipeline-stage ${badgeClass}" title="${s.key}: ${status}">
            ${s.label}
          </span>
        `;
      }).join('<span class="pipeline-connector">&rarr;</span>')}
    </div>
  `;
}

function openWorkflowAuditDrawer(run) {
  state.openDrawer({
    title: `Workflow Audit: ${run.run_id.substring(0, 16)}...`,
    subtitle: `Trigger: ${run.trigger_type} • Status: ${run.status.toUpperCase()}`,
    renderContent: (el) => {
      const records = run.stage_records || {};

      el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:20px;">
          <!-- Execution Overview -->
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; background:var(--bg-canvas); padding:14px; border-radius:8px;">
            <div>
              <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Target Stop Stage</div>
              <div style="font-family:var(--font-mono); font-size:12px; margin-top:2px;">${state.escapeHtml(run.target_stop_stage)}</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Correlation ID</div>
              <div style="display:flex; align-items:center; gap:8px; margin-top:2px;">
                <span style="font-family:var(--font-mono); font-size:12px; color:var(--accent-cyan);">${state.escapeHtml(run.correlation_id || 'N/A')}</span>
                ${run.correlation_id ? `
                  <button class="btn btn-ghost btn-sm" id="btn-drawer-copy-corr" style="padding:1px 6px; font-size:10px;" title="Copy Correlation ID">
                    📋 Copy
                  </button>
                ` : ''}
              </div>
            </div>
          </div>

          <!-- Stage Telemetry Cards -->
          <div>
            <div class="form-label" style="margin-bottom:10px;">Executed Stage Telemetry</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
              ${Object.entries(records).map(([sKey, rec]) => {
                const outMap = rec.output_ids || {};
                return `
                <div style="background:var(--bg-canvas); border:1px solid ${rec.status === 'failed' ? 'var(--status-danger-border)' : 'var(--border-subtle)'}; border-radius:8px; padding:12px 14px;">
                  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-family:var(--font-mono); font-weight:600; font-size:12px; color:var(--text-primary);">${state.escapeHtml(sKey)}</span>
                    <span class="badge ${rec.status === 'completed' ? 'badge-success' : rec.status === 'failed' ? 'badge-danger' : 'badge-neutral'}">
                      ${state.escapeHtml(rec.status)}
                    </span>
                  </div>

                  <div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px; font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
                    <span>Duration: ${rec.duration_ms ? rec.duration_ms.toFixed(1) + 'ms' : 'N/A'}</span>
                    <span>&bull;</span>
                    <span>Attempts: ${rec.attempt_count || 1}</span>
                    ${outMap.retrieval_mode ? `<span>&bull;</span><span class="badge badge-neutral" style="font-size:10px;">Mode: ${state.escapeHtml(outMap.retrieval_mode)}</span>` : ''}
                    ${outMap.routing_strategy ? `<span>&bull;</span><span class="badge badge-cyan" style="font-size:10px;">Route: ${state.escapeHtml(outMap.routing_strategy)}</span>` : ''}
                    ${outMap.retrieved_chunks_count !== undefined ? `<span>&bull;</span><span>Chunks: ${outMap.retrieved_chunks_count}</span>` : ''}
                  </div>

                  ${rec.error ? `
                    <div style="margin-top:8px; padding:8px 10px; background:var(--status-danger-bg); border-left:3px solid var(--status-danger); border-radius:4px; font-size:11px; color:var(--status-danger); font-family:var(--font-mono);">
                      <strong>[${state.escapeHtml(rec.error.error_category || 'ERROR')}]</strong>: ${state.escapeHtml(rec.error.error_message || JSON.stringify(rec.error))}
                    </div>
                  ` : ''}
                </div>
              `;
              }).join('')}
            </div>
          </div>


          <!-- Raw Checkpointed Outputs -->
          <div>
            <div class="form-label" style="margin-bottom:8px;">Produced Domain Entity Output IDs</div>
            <div class="json-inspector">${state.escapeHtml(JSON.stringify(run.produced_outputs || {}, null, 2))}</div>
          </div>
        </div>
      `;

      el.querySelector('#btn-drawer-copy-corr')?.addEventListener('click', () => {
        if (run.correlation_id) {
          navigator.clipboard.writeText(run.correlation_id);
          state.showToast('Correlation ID copied to clipboard', 'info');
        }
      });
    },
    renderFooter: () => `
      <button class="btn btn-secondary btn-sm" onclick="window.state.closeDrawer()">Close</button>
    `,

  });
}

function openDispatchWorkflowModal(workspaceId, competitors, contexts, onSuccess) {
  state.openModal({
    title: 'Dispatch Operational Workflow',
    renderBody: (el) => {
      el.innerHTML = `
        <form id="form-dispatch-workflow" style="display:flex; flex-direction:column; gap:16px;">
          <div class="form-group">
            <label class="form-label">Competitor Company *</label>
            <select class="form-control" id="wf-comp" required>
              ${competitors.map(c => `<option value="${c.competitor_id}">${state.escapeHtml(c.name)}</option>`).join('')}
            </select>
          </div>

          <div class="form-group">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <label class="form-label">Raw Acquired Document Text *</label>
              <div style="display:flex; gap:6px;">
                <button type="button" class="btn btn-ghost btn-sm" id="btn-preset-pricing" style="font-size:11px; padding:2px 6px;">Pricing Preset</button>
                <button type="button" class="btn btn-ghost btn-sm" id="btn-preset-product" style="font-size:11px; padding:2px 6px;">Product Preset</button>
              </div>
            </div>
            <textarea class="form-control" id="wf-content" placeholder="Paste extracted document text, pricing table, or press preset buttons..." style="min-height:120px;" required></textarea>
          </div>

          <div class="form-group">
            <label class="form-label">Origin Source URL</label>
            <input type="url" class="form-control" id="wf-url" placeholder="https://competitor.com/pricing" />
          </div>

          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div class="form-group">
              <label class="form-label">Target Stop Stage</label>
              <select class="form-control" id="wf-stop-stage">
                <option value="knowledge_processing">Knowledge Ingestion Only</option>
                <option value="retrieval">Stop at Retrieval</option>
                <option value="event_analysis">Stop at Event Analysis</option>
                <option value="signal_detection">Stop at Signal Detection</option>
                <option value="insight_generation" selected>Insight Generation (Default)</option>
                <option value="decision_support">Full Pipeline + Decision Support</option>
              </select>
            </div>

            <div class="form-group">
              <label class="form-label">Decision Context (If DS Enabled)</label>
              <select class="form-control" id="wf-context">
                <option value="">None / Auto</option>
                ${contexts.map(ctx => `<option value="${ctx.decision_id}">${state.escapeHtml(ctx.title)}</option>`).join('')}
              </select>
            </div>
          </div>
        </form>
      `;

      // Preset Loaders
      el.querySelector('#btn-preset-pricing')?.addEventListener('click', () => {
        const textarea = el.querySelector('#wf-content');
        const urlInput = el.querySelector('#wf-url');
        if (textarea) textarea.value = 'ACME Cloud Platform announced sweeping price changes today. The Enterprise Tier is reduced from $99/mo to $49/mo, with unlimited API requests and dedicated cluster hosting included. A new Developer Starter tier is now free for up to 100,000 monthly transactions.';
        if (urlInput) urlInput.value = 'https://acme.example.com/pricing-announcement';
      });

      el.querySelector('#btn-preset-product')?.addEventListener('click', () => {
        const textarea = el.querySelector('#wf-content');
        const urlInput = el.querySelector('#wf-url');
        if (textarea) textarea.value = 'Release 4.0 Launch: Today we are introducing real-time agentic reasoning, native Vector search integration, and multi-tenant VPC peering. Deprecating legacy REST v1 API by end of Q4.';
        if (urlInput) urlInput.value = 'https://acme.example.com/changelog/v4-launch';
      });
    },
    renderFooter: () => `
      <button class="btn btn-ghost" onclick="window.state.closeModal()">Cancel</button>
      <button class="btn btn-primary" id="btn-submit-workflow">Dispatch Workflow</button>
    `,
  });

  document.getElementById('btn-submit-workflow')?.addEventListener('click', async () => {
    const compId = document.getElementById('wf-comp')?.value;
    const content = document.getElementById('wf-content')?.value.trim();
    const sourceUrl = document.getElementById('wf-url')?.value.trim() || 'https://monitored-source.local';
    const stopStage = document.getElementById('wf-stop-stage')?.value;
    const contextId = document.getElementById('wf-context')?.value;

    if (!compId || !content) {
      state.showToast('Competitor and document text are required', 'warning');
      return;
    }

    const btn = document.getElementById('btn-submit-workflow');
    if (btn) btn.disabled = true;

    try {
      const payload = {
        company_id: compId,
        raw_document_content: content,
        source_url: sourceUrl,
        target_stop_stage: stopStage,
        trigger_type: 'manual_user_request',
        enable_decision_support: stopStage === 'decision_support',
      };

      if (contextId) {
        const matched = contexts.find(c => c.decision_id === contextId);
        if (matched) {
          payload.decision_context = matched;
        }
      }

      const run = await api.submitWorkflowRun(workspaceId, payload);
      state.showToast(`Workflow dispatched [${run.status}]: ${run.run_id.substring(0, 12)}...`, 'success');
      state.closeModal();
      if (onSuccess) onSuccess();
    } catch (err) {
      state.showToast(err.message, 'error');
      if (btn) btn.disabled = false;
    }
  });
}
