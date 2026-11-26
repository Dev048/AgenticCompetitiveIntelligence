---
title: Competitive Intelligence Agent
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Evidence-grounded agentic RAG for competitive intelligence
---

# Competitive Intelligence Agent: live demo

A read-only demo workspace with **fictional data** (three invented competitors in a
product-analytics market), seeded through the real pipeline on every boot.

What to try:

1. **Strategic Insights → Actionable Recommendations → "Trace to sources"**: follow a
   `PROPOSED` recommendation down 10 provenance layers to the exact quote, the document
   version's SHA-256 hash and the source URL.
2. **Ask the Evidence** in **Agentic** mode: *"Does the Northwind Free plan really include
   3 seats?"* shows query analysis, sub-queries, verification and a detected contradiction.
3. **Approve or reject** a recommendation: the human decision boundary.

Writes other than queries and recommendation review are disabled, and live queries are
rate-limited. If the free LLM quota is exhausted, queries fall back to deterministic
retrieval and say so.

Source code, architecture and how this demo is built: see the project's GitHub repository.
