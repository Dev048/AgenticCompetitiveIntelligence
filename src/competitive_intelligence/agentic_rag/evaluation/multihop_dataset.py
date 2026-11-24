"""Deterministic Multi-Hop and Adversarial Benchmark Dataset.

Contains a frozen corpus of 115 curated domain chunks and 50 benchmark queries
spanning 9 distinct categories with explicit ground-truth relevance grades (0-3).
"""

from datetime import datetime, timezone
import uuid

from competitive_intelligence.agentic_rag.evaluation.multihop_models import (
    BenchmarkCategory,
    MultiHopEvaluationDataset,
    MultiHopEvaluationQuery,
)
from competitive_intelligence.agentic_rag.models import RetrievalDecision
from competitive_intelligence.knowledge_pipeline.models import ChunkMetadata, DocumentChunk
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash


def _build_chunk(
    chunk_id: str,
    content: str,
    company_id: str,
    source_category: str,
    doc_id: str | None = None,
    version_id: str = "v1",
) -> DocumentChunk:
    """Helper to construct a deterministic DocumentChunk with fixed identifiers."""
    d_id = doc_id or f"doc_{chunk_id.replace('chunk_', '')}"
    meta = ChunkMetadata(
        source_id="src_multihop_benchmark",
        document_id=d_id,
        version_id=version_id,
        company_id=company_id,
        workspace_id="ws_multihop_eval",
        source_url=f"https://benchmark.ci.internal/docs/{company_id}/{d_id}",
        source_category=source_category,
        collected_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=0,
        total_chunks=1,
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=d_id,
        version_id=version_id,
        content=content,
        content_hash=compute_content_hash(content),
        metadata=meta,
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


def get_multihop_evaluation_dataset() -> MultiHopEvaluationDataset:
    """Construct and return the frozen Multi-Hop Evaluation Dataset (115 chunks, 50 queries)."""
    
    # -------------------------------------------------------------------------
    # 1. Controlled Evaluation Corpus (115 Chunks)
    # -------------------------------------------------------------------------
    raw_corpus_definitions = [
        # Snowflake Architecture, Models, and Pricing
        ("chunk_001", "Snowflake Arctic is an enterprise-focused open LLM featuring a 480B Mixture-of-Experts architecture with 17B active parameters, optimized for SQL generation and code completion under Apache 2.0.", "snowflake", "news", "doc_sf_arctic"),
        ("chunk_002", "Snowflake Arctic benchmark evaluations show 78.5% on Spider SQL benchmark and 64.2% on HumanEval, rivaling larger proprietary models while keeping inference costs low.", "snowflake", "technical", "doc_sf_arctic"),
        ("chunk_003", "Snowflake Arctic v2 preview announcement highlights a dense-MoE hybrid router designed for multi-table join synthesis and sub-second warehouse planning.", "snowflake", "news", "doc_sf_arctic_v2", "v2"),
        ("chunk_004", "Snowflake Cortex AI pricing charges 0.42 credits per 1,000 prompt tokens for fine-tuned Arctic models running on standard virtual warehouses.", "snowflake", "pricing", "doc_sf_cortex"),
        ("chunk_005", "Snowflake Cortex AI regional availability includes AWS us-east-1 and Azure East US, offering 99.9% uptime with 4-hour RTO and 15-minute RPO disaster recovery commitments.", "snowflake", "architecture", "doc_sf_cortex_sla"),
        ("chunk_006", "Snowflake announces FedRAMP High authorization across all AWS government regions, enabling federal agencies to process classified analytics.", "snowflake", "compliance", "doc_sf_fedramp"),
        ("chunk_007", "Snowflake Data Clean Rooms allow cross-company data sharing and monetization without exposing underlying PII, governed via differential privacy policies.", "snowflake", "product", "doc_sf_cleanrooms"),
        ("chunk_008", "Benchmark tests show Snowflake Arctic inference on NVIDIA H100 achieves 142 tokens/second per node, consuming 2.2 warehouse credits per query hour.", "snowflake", "technical", "doc_sf_perf"),
        ("chunk_009", "Snowflake sponsored study argues that Snowflake Lakehouse engine is 1.8x faster and 35% cheaper than Databricks on the standard TPC-DS 100TB workload.", "snowflake", "blog", "doc_sf_tpcds"),

        # Databricks DBRX, MosaicML, Tabular Acquisition, Pricing
        ("chunk_010", "Databricks released DBRX, a 132B parameter MoE open model with 36B active parameters, trained on 12T tokens of data and licensed under Databricks Open Model License.", "databricks", "news", "doc_db_dbrx"),
        ("chunk_011", "Databricks DBRX achieves 70.1% on HumanEval coding benchmark and 66.9% on GSM8k reasoning benchmark, surpassing Llama 2 70B.", "databricks", "technical", "doc_db_dbrx"),
        ("chunk_012", "Databricks acquired Tabular for $1.5 billion in June 2024 to unite Apache Iceberg with Linux Foundation Delta Lake in a single storage format.", "databricks", "news", "doc_db_tabular_acq"),
        ("chunk_013", "Databricks roadmap states that the Tabular metadata engine will be natively unified into Unity Catalog by Q4 2024, enabling zero-copy Iceberg queries.", "databricks", "roadmap", "doc_db_tabular_roadmap"),
        ("chunk_014", "Databricks MosaicML team is aggressively hiring LLM compiler engineers and distributed cluster architects in San Francisco to scale training infra.", "databricks", "careers", "doc_db_careers"),
        ("chunk_015", "Databricks Serverless Compute pricing starts at $0.07 per DBU for standard SQL workloads with auto-scaling down to zero in 5 minutes.", "databricks", "pricing", "doc_db_pricing"),
        ("chunk_016", "Databricks Unity Catalog provides centralized access control, lineage tracking, and audit log retention configured for 365 days across AWS, Azure, and GCP.", "databricks", "architecture", "doc_db_unity"),
        ("chunk_017", "Databricks published benchmark asserts that Lakehouse SQL execution is 2.5x lower cost and 1.4x faster than Snowflake on TPC-DS 30TB benchmarks.", "databricks", "blog", "doc_db_tpcds"),

        # OpenAI Pricing, GPT-4o, Rockset, Enterprise & Compliance
        ("chunk_018", "OpenAI GPT-4o API pricing is set at $5.00 per 1M input tokens and $15.00 per 1M output tokens, representing a 50% price reduction compared to GPT-4 Turbo.", "openai", "pricing", "doc_oai_gpt4o_price"),
        ("chunk_019", "OpenAI GPT-4 Turbo (gpt-4-0125-preview) pricing was $10.00 per 1M input tokens and $30.00 per 1M output tokens with a 128k context window.", "openai", "pricing", "doc_oai_gpt4t_price", "v0"),
        ("chunk_020", "OpenAI Batch API offers a 50% cost reduction for non-realtime asynchronous requests processed within a 24-hour turnaround window.", "openai", "pricing", "doc_oai_batch"),
        ("chunk_021", "OpenAI Enterprise tier provides SOC2 Type II compliance, zero data retention for model training, dedicated tenant isolation, and custom data residency.", "openai", "compliance", "doc_oai_compliance"),
        ("chunk_022", "OpenAI is hiring Systems Software Engineers for supercomputing infrastructure in San Francisco to optimize multi-datacenter GPU cluster fabrics.", "openai", "careers", "doc_oai_careers"),
        ("chunk_023", "OpenAI acquired Rockset in June 2024 to integrate its real-time indexing and search engine into enterprise ChatGPT and vector retrieval features.", "openai", "news", "doc_oai_rockset"),
        ("chunk_024", "Following the Rockset acquisition, OpenAI integrated distributed inverted indexes, reducing enterprise hybrid search p95 query latency from 85ms to 18ms.", "openai", "technical", "doc_oai_rockset_tech"),
        ("chunk_025", "OpenAI Enterprise SLA guarantees 99.9% monthly uptime, providing a 10% credit refund for downtime below 99.9% and 30% credit refund for downtime below 99.0%.", "openai", "sla", "doc_oai_sla"),
        ("chunk_026", "OpenAI marketing claims that GPT-4o real-time voice mode achieves an average voice-to-voice latency of 232ms, comparable to human conversation.", "openai", "marketing", "doc_oai_voice_mktg"),
        ("chunk_027", "Independent developer API measurements report that OpenAI GPT-4o real-time voice API exhibits an average roundtrip latency of 650ms under peak load.", "openai", "technical", "doc_oai_voice_eval"),

        # Anthropic Claude 3.0 vs 3.5, Pricing, Alignment Lab
        ("chunk_028", "Anthropic Claude 3.5 Sonnet API pricing is $3.00 per 1M input tokens and $15.00 per 1M output tokens, with 200k context window and 8192 max output tokens.", "anthropic", "pricing", "doc_ant_35_price"),
        ("chunk_029", "Anthropic Claude 3.0 Opus API pricing was $15.00 per 1M input tokens and $75.00 per 1M output tokens, achieving 84.9% on HumanEval coding benchmark.", "anthropic", "pricing", "doc_ant_30_price", "v0"),
        ("chunk_030", "Anthropic Claude 3.5 Sonnet achieves 92.0% on 0-shot HumanEval and 64.0% on SWE-bench Verified, outperforming Claude 3.0 Opus at one-fifth the cost.", "anthropic", "technical", "doc_ant_35_eval"),
        ("chunk_031", "Anthropic Message Batches API offers a 50% discount on Claude 3.5 Sonnet processing for asynchronous workflows completed within 24 hours.", "anthropic", "pricing", "doc_ant_batches"),
        ("chunk_032", "Anthropic established a dedicated Alignment Research lab in London, hiring Mechanistic Interpretability researchers to decode model internal circuits.", "anthropic", "careers", "doc_ant_london_lab"),
        ("chunk_033", "Anthropic's mechanistic interpretability team extracted millions of monosemantic features from Claude 3.0 Sonnet, mapping concepts to specific neuron activations.", "anthropic", "technical", "doc_ant_interpretability"),
        ("chunk_034", "Anthropic API Tier 4 organizations receive rate limits of 400,000 tokens per minute (TPM) and 4,000 requests per minute (RPM) for Claude 3.5 Sonnet.", "anthropic", "sla", "doc_ant_rate_limits"),

        # Google Gemini 1.0 vs 1.5, Context, Pricing, Careers
        ("chunk_035", "Google Gemini 1.5 Pro features a 2 million token context window, priced at $3.50 per 1M input tokens for prompts up to 128k and $7.00 per 1M for prompts over 128k.", "google", "pricing", "doc_goog_15_price"),
        ("chunk_036", "Google Gemini 1.0 Ultra supported a 32,000 token context window, priced at $10.00 per 1M input tokens and $30.00 per 1M output tokens.", "google", "pricing", "doc_goog_10_price", "v0"),
        ("chunk_037", "Google Gemini 1.5 Flash is priced at $0.075 per 1M input tokens and $0.30 per 1M output tokens, optimized for high-volume low-latency tasks.", "google", "pricing", "doc_goog_flash_price"),
        ("chunk_038", "Google DeepMind expanded its London reinforcement learning team, recruiting research scientists to work on Gemini self-improving reasoning loops.", "google", "careers", "doc_goog_careers"),
        ("chunk_039", "Google Workspace offers Gemini Enterprise add-on for $30 per user per month, integrating AI document drafting and meeting summaries in Google Meet.", "google", "pricing", "doc_goog_workspace"),
        ("chunk_040", "Google Vertex AI provisioned throughput offers guaranteed QPS capacity for Gemini models with annual commitments and 99.95% availability SLA.", "google", "pricing", "doc_goog_vertex_sla"),

        # Microsoft Copilot, Azure, Windows
        ("chunk_041", "Microsoft Copilot for Microsoft 365 Enterprise is priced at $30 per user per month with an annual commitment, requiring Microsoft 365 E3 or E5.", "microsoft", "pricing", "doc_msft_copilot"),
        ("chunk_042", "GitHub Copilot Enterprise is priced at $39 per user per month, offering codebase indexing, custom pull request summaries, and Bing integration.", "microsoft", "pricing", "doc_msft_gh_copilot"),
        ("chunk_043", "Azure OpenAI Service provides HIPAA and FedRAMP High compliance within Microsoft's sovereign enterprise cloud boundaries.", "microsoft", "compliance", "doc_msft_compliance"),

        # Qdrant Vector Search, Storage Optimization, Quantization
        ("chunk_044", "Qdrant vector database implements scalar quantization and 1-bit binary quantization, reducing memory footprint by up to 97% with minimal recall loss.", "qdrant", "technical", "doc_qdrant_quant"),
        ("chunk_045", "Qdrant uses disk-backed payload paging with scalar quantization, storing vector payloads on NVMe SSDs to optimize cold storage cost efficiency.", "qdrant", "technical", "doc_qdrant_storage"),
        ("chunk_046", "Qdrant native hybrid search combines dense embeddings with sparse BM25 vectors, using Reciprocal Rank Fusion (RRF) with configurable alpha weighting.", "qdrant", "technical", "doc_qdrant_hybrid"),
        ("chunk_047", "Qdrant sparse BM25 indexing introduces a 1.35x RAM multiplier over raw dense indexes during active in-memory graph construction.", "qdrant", "technical", "doc_qdrant_sparse_ram"),
        ("chunk_048", "Qdrant Cloud provides SOC2 Type II compliance, automated zero-downtime cluster upgrades, and tenant-level encryption at rest.", "qdrant", "compliance", "doc_qdrant_cloud"),
        ("chunk_049", "Qdrant official documentation reports 2.4 GB memory consumption per 1M 1536-dimensional vectors using scalar quantization and HNSW m=16.", "qdrant", "technical", "doc_qdrant_mem_doc"),
        ("chunk_050", "Community load test report shows Qdrant consuming 4.1 GB memory per 1M 1536-dim vectors during high concurrent write workloads of 5,000 inserts/sec.", "qdrant", "technical", "doc_qdrant_mem_bench"),

        # Pinecone Vector Search, Serverless, Benchmarks
        ("chunk_051", "Pinecone Serverless separates read, write, and storage tiers, utilizing blob storage for long-term index persistence and per-request compute billing.", "pinecone", "architecture", "doc_pine_serverless"),
        ("chunk_052", "Pinecone Serverless whitepaper claims p99 query latency of 12ms under 10,000 queries per second on 1536-dimensional OpenAI embeddings.", "pinecone", "marketing", "doc_pine_whitepaper"),
        ("chunk_053", "Independent vector benchmark report measures Pinecone Serverless p99 query latency at 45ms under 10,000 QPS with p99 cold-start spikes exceeding 180ms.", "pinecone", "technical", "doc_pine_bench"),
        ("chunk_054", "Pinecone enterprise support SLA guarantees 99.99% availability with dedicated 15-minute response times for critical severity outages.", "pinecone", "sla", "doc_pine_sla"),

        # Meta Llama 3 & Licensing
        ("chunk_055", "Meta released Llama 3 in 8B and 70B parameter configurations, trained on 15T tokens with a context window of 8,192 tokens.", "meta", "news", "doc_meta_llama3"),
        ("chunk_056", "Meta Llama 3 Community License allows free commercial use up to 700 million monthly active users, beyond which entities must request a custom commercial license.", "meta", "compliance", "doc_meta_license"),
        ("chunk_057", "Meta Llama 3 70B achieved 82.0% on MMLU and 81.7% on HumanEval, establishing state-of-the-art results among publicly available open-weights models.", "meta", "technical", "doc_meta_eval"),

        # Mistral AI Models, Pricing, Careers, Benchmarks
        ("chunk_058", "Mistral AI announced Mistral Large 2 (123B parameters) with a 128k context window and advanced code generation capabilities across 80+ languages.", "mistral", "news", "doc_mis_large2"),
        ("chunk_059", "Mistral Large 2 API pricing is $3.00 per 1M input tokens and $9.00 per 1M output tokens through La Plateforme.", "mistral", "pricing", "doc_mis_pricing"),
        ("chunk_060", "Mistral NeMo was released with a 12B parameter size and a 128,000 token context window in partnership with NVIDIA.", "mistral", "news", "doc_mis_nemo"),
        ("chunk_061", "Mistral AI is hiring Distributed Training Engineers in Paris to design ultra-large scale mixture-of-experts parallelization topologies.", "mistral", "careers", "doc_mis_careers"),
        ("chunk_062", "Mistral AI release notes report that Mistral Large 2 achieves 85 tokens per second generation speed on 8x NVIDIA H100 80GB SXM5 nodes.", "mistral", "marketing", "doc_mis_speed_official"),
        ("chunk_063", "Third-party cloud hosting benchmark reports Mistral Large 2 achieving 42 tokens per second generation throughput under standard production concurrency.", "mistral", "technical", "doc_mis_speed_eval"),

        # xAI Grok, Supercluster
        ("chunk_064", "xAI open-sourced Grok-1, a 314 billion parameter Mixture-of-Experts model with 86B active parameters under the Apache 2.0 license.", "xai", "news", "doc_xai_grok1"),
        ("chunk_065", "xAI constructed the Memphis Colossus supercluster featuring 100,000 liquid-cooled NVIDIA H100 GPUs powered by a 150MW dedicated power substation.", "xai", "news", "doc_xai_memphis"),
        ("chunk_066", "xAI engineering documentation targets a 1.2 trillion parameter dense-MoE hybrid model architecture for Grok-3 trained on the Memphis cluster.", "xai", "roadmap", "doc_xai_grok3"),

        # Inference Engines: vLLM, TensorRT-LLM, HuggingFace
        ("chunk_067", "vLLM v0.3.0 introduced basic PagedAttention supporting continuous batching with 1.4x higher throughput than HuggingFace TGI.", "vllm", "technical", "doc_vllm_v03", "v0"),
        ("chunk_068", "vLLM v0.4.2 added chunked prefill and draft model speculative decoding, boosting multi-turn chat throughput by 2.8x over v0.3.0.", "vllm", "technical", "doc_vllm_v04", "v1"),
        ("chunk_069", "Speculative token generation acceleration in vLLM pairs a small draft model with a large target model to verify multiple candidate tokens in a single forward pass.", "vllm", "technical", "doc_vllm_speculative"),

        # Distractors / Adversarial / High-Overlap Chunks
        ("chunk_070", "Apple Intelligence introduces on-device 3B small language models and Private Cloud Compute for iOS, iPadOS, and macOS, with zero support for Android devices.", "apple", "news", "doc_apple_intel"),
        ("chunk_071", "Agricultural telemetry report on snowflake crop refrigeration systems describes sub-zero cooling efficiency in northern apple farming storage silos.", "agriculture_inc", "general", "doc_distractor_agri_sf"),
        ("chunk_072", "South American textile report details Peruvian llama wool grading standards, focusing on 70B fiber tensile strength and high-density fleece exports.", "textile_global", "general", "doc_distractor_llama_wool"),
        ("chunk_073", "SQLite vector extension documentation explains in-memory brute-force cosine distance search for embedded systems without GPU hardware acceleration.", "sqlite_org", "technical", "doc_distractor_sqlite"),
        ("chunk_074", "AWS Bedrock supports models from Anthropic, Meta, Mistral, Cohere, and Amazon Titan, but does not offer Google Gemini or OpenAI GPT models on its platform.", "amazon", "news", "doc_distractor_bedrock_goog"),
        ("chunk_075", "Merger analysis clarifies that OpenAI and Databricks remain independent rival corporations with zero acquisition, merger, or joint equity agreements.", "tech_review", "news", "doc_distractor_oai_db"),

        # Additional domain chunks for breadth (076 - 115)
        ("chunk_076", "Cohere Command R+ is an enterprise LLM with 104B parameters optimized for multilingual RAG and tool use across 10 business languages.", "cohere", "news", "doc_cohere_cmd"),
        ("chunk_077", "Cohere Command R+ pricing is $3.00 per 1M input tokens and $15.00 per 1M output tokens on Cohere Cloud and AWS Bedrock.", "cohere", "pricing", "doc_cohere_price"),
        ("chunk_078", "Together AI launched Mixture-of-Agents platform achieving 65.1% on AlpacaEval 2.0 by aggregating responses from multiple open source LLMs.", "together", "news", "doc_together_moa"),
        ("chunk_079", "DeepSeek Coder V2 supports 338 programming languages with 236B total parameters (21B active) under an open commercial license.", "deepseek", "news", "doc_deepseek_v2"),
        ("chunk_080", "DeepSeek API pricing is set at $0.14 per 1M input tokens and $0.28 per 1M output tokens, undercutting Western LLM providers by over 90%.", "deepseek", "pricing", "doc_deepseek_price"),
        ("chunk_081", "Amazon Bedrock provisioned throughput allows purchasing custom model capacity units (Model Units) on 1-month or 6-month commitments.", "amazon", "pricing", "doc_aws_provisioned"),
        ("chunk_082", "Microsoft Azure AI Search integrates vector search, hybrid ranking, and semantic reranking powered by Bing models.", "microsoft", "product", "doc_azure_search"),
        ("chunk_083", "Scale AI announced Defense Llama evaluation framework tailored for national security and military operational planning.", "scale", "news", "doc_scale_defense"),
        ("chunk_084", "HuggingFace Transformers v4.42 added native support for Gemma 2 and quantized KV cache compression.", "huggingface", "technical", "doc_hf_v442"),
        ("chunk_085", "LangChain v0.3 introduced strict separation of core runtime and integration libraries with improved async streaming support.", "langchain", "technical", "doc_lc_v03"),
        ("chunk_086", "LlamaIndex v0.11 added workflows engine with event-driven state orchestration for complex multi-agent pipelines.", "llamaindex", "technical", "doc_li_v011"),
        ("chunk_087", "Milvus 2.4 vector database introduced GPU-accelerated indexing with NVIDIA CAGRA and multi-vector search support.", "zilliz", "technical", "doc_milvus_24"),
        ("chunk_088", "Weaviate vector database introduced dynamic index switching between HNSW and flat indexing based on tenant collection size.", "weaviate", "technical", "doc_weaviate_dyn"),
        ("chunk_089", "Chroma vector database release 0.5.0 added distributed clustering on Kubernetes and Rust-based query execution engine.", "chroma", "technical", "doc_chroma_05"),
        ("chunk_090", "NVIDIA TensorRT-LLM v0.11 supports FP4 quantization on Blackwell architectures, doubling inference throughput per GPU node.", "nvidia", "technical", "doc_nv_trt"),
        ("chunk_091", "NVIDIA announced NIM (NVIDIA Inference Microservices) providing containerized self-hosted LLM endpoints with optimized CUDA kernels.", "nvidia", "news", "doc_nv_nim"),
        ("chunk_092", "Apple Private Cloud Compute runs on custom Apple Silicon servers with hardware-enforced memory encryption and verifiable transparency.", "apple", "technical", "doc_apple_pcc"),
        ("chunk_093", "Anyscale Endpoints reduced Ray Serve cold start latency by 60% through pre-warmed container snapshots and warm GPU caching.", "anyscale", "technical", "doc_anyscale_cold"),
        ("chunk_094", "Fireworks AI announced fireattention kernel optimizing multi-head latent attention for DeepSeek V2 models at 120 tokens/sec.", "fireworks", "technical", "doc_fw_fireatt"),
        ("chunk_095", "Groq LPU inference engine delivers 550 tokens per second for Llama 3 8B using deterministic tensor streaming architecture.", "groq", "technical", "doc_groq_lpu"),
        ("chunk_096", "Cerebras CS-3 wafer-scale engine delivers 1800 tokens per second for Llama 3 70B, claiming 20x speedup over NVIDIA H100 clusters.", "cerebras", "technical", "doc_cerebras_cs3"),
        ("chunk_097", "SambaNova SN40L reconfigurable dataflow unit delivers high-efficiency 1000 tokens/sec on 100B+ mixture of experts models.", "sambanova", "technical", "doc_samba_sn40"),
        ("chunk_098", "Replicate introduced serverless fine-tuning endpoints allowing custom LoRA training on Llama 3 in under 10 minutes.", "replicate", "product", "doc_rep_lora"),
        ("chunk_099", "Modal Labs serverless cloud added instant GPU cold starts from 0 to live PyTorch container in 1.2 seconds.", "modal", "product", "doc_modal_gpu"),
        ("chunk_100", "Baseten introduced model autoscaling with zero-scale cold boot latency of 800ms using flash-injected model weights.", "baseten", "product", "doc_base_auto"),
        ("chunk_101", "OctoAI was acquired by NVIDIA in 2024 to strengthen NVIDIA's cloud inference compilation and edge optimization tooling.", "nvidia", "news", "doc_nv_octo"),
        ("chunk_102", "Lambda Labs expanded its on-demand H100 cloud cluster capacity with a new 50MW data center in Texas.", "lambda", "news", "doc_lambda_h100"),
        ("chunk_103", "Crusoe Energy deployed 10,000 GPUs on methane flare mitigation data centers in North Dakota for renewable AI training.", "crusoe", "news", "doc_crusoe_green"),
        ("chunk_104", "CoreWeave secured $7.5B in debt financing to expand its Tier III data center footprint across North America and Europe.", "coreweave", "news", "doc_cw_debt"),
        ("chunk_105", "RunPod launched serverless vLLM endpoints with per-second billing and automatic multi-GPU tensor parallel scaling.", "runpod", "product", "doc_runpod_vllm"),
        ("chunk_106", "Perplexity AI launched Perplexity Enterprise Pro for $40/user/month with SOC2 Type II compliance and SSO integration.", "perplexity", "pricing", "doc_perp_ent"),
        ("chunk_107", "You.com launched Custom Assistants platform allowing enterprises to connect private knowledge bases and API actions.", "you_com", "product", "doc_you_custom"),
        ("chunk_108", "Harvey AI secured $100M Series C led by Google Ventures to build specialized legal intelligence models on Claude 3.5.", "harvey", "news", "doc_harvey_c"),
        ("chunk_109", "Glean enterprise search raised $200M at a $4.6B valuation to expand multi-modal workplace search connectors.", "glean", "news", "doc_glean_val"),
        ("chunk_110", "Hebbia raised $130M Series B led by Andreessen Horowitz to automate financial and legal due diligence using multi-agent search.", "hebbia", "news", "doc_hebbia_b"),
        ("chunk_111", "Writer Inc released Palmyra X 004 model achieving state-of-the-art tool-calling accuracy for enterprise workflow automation.", "writer", "news", "doc_writer_palmyra"),
        ("chunk_112", "Deci AI was acquired by NVIDIA in 2024 to integrate its Neural Architecture Search (NAS) into TensorRT optimization workflows.", "nvidia", "news", "doc_nv_deci"),
        ("chunk_113", "Character.AI signed a non-exclusive technology licensing agreement with Google, with founders rejoining Google DeepMind.", "google", "news", "doc_cai_google"),
        ("chunk_114", "Adept AI founders joined Amazon to lead AGI agent engineering, licensing Adept's multi-modal automation technology to AWS.", "amazon", "news", "doc_adept_amazon"),
        ("chunk_115", "Inflection AI transitioned to an enterprise AI studio model after Microsoft hired its core team and licensed its Inflection-2.5 architecture.", "microsoft", "news", "doc_inflection_msft"),
    ]

    corpus: list[DocumentChunk] = []
    for item in raw_corpus_definitions:
        cid = item[0]
        text = item[1]
        comp = item[2]
        cat = item[3]
        doc = item[4]
        ver = item[5] if len(item) > 5 else "v1"
        corpus.append(_build_chunk(chunk_id=cid, content=text, company_id=comp, source_category=cat, doc_id=doc, version_id=ver))

    # -------------------------------------------------------------------------
    # 2. Frozen Evaluation Queries (50 Queries Across 9 Categories)
    # -------------------------------------------------------------------------
    queries: list[MultiHopEvaluationQuery] = [
        # Category 1: Multi-Entity Comparison (6 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_01",
            query_text="Compare Snowflake Arctic and Databricks DBRX architecture, SQL generation benchmark, and licensing terms.",
            category=BenchmarkCategory.MULTI_ENTITY,
            relevant_chunk_ids={"chunk_001": 3, "chunk_002": 3, "chunk_010": 3, "chunk_011": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=2,
            expected_subqueries_max=3,
            notes="Requires decomposing into Snowflake Arctic architecture/benchmarks and Databricks DBRX architecture/benchmarks.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_02",
            query_text="Compare OpenAI GPT-4o and Anthropic Claude 3.5 Sonnet pricing per 1M tokens for input and output.",
            category=BenchmarkCategory.MULTI_ENTITY,
            relevant_chunk_ids={"chunk_018": 3, "chunk_028": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=2,
            expected_subqueries_max=2,
            notes="Comparative query across OpenAI and Anthropic pricing.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_03",
            query_text="Compare Qdrant and Pinecone vector database memory optimization, quantization, and serverless architectures.",
            category=BenchmarkCategory.MULTI_ENTITY,
            relevant_chunk_ids={"chunk_044": 3, "chunk_045": 3, "chunk_051": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=2,
            expected_subqueries_max=3,
            notes="Requires gathering quantization info from Qdrant and serverless architecture from Pinecone.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_04",
            query_text="Compare Google Gemini 1.5 Pro and Anthropic Claude 3.5 Sonnet context window sizes and coding benchmark scores.",
            category=BenchmarkCategory.MULTI_ENTITY,
            relevant_chunk_ids={"chunk_035": 3, "chunk_028": 2, "chunk_030": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=2,
            expected_subqueries_max=2,
            notes="Multi-entity comparison of long-context models and coding benchmarks.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_05",
            query_text="Compare Microsoft Copilot for M365 Enterprise pricing with Google Workspace Gemini Enterprise add-on pricing.",
            category=BenchmarkCategory.MULTI_ENTITY,
            relevant_chunk_ids={"chunk_041": 3, "chunk_039": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=2,
            expected_subqueries_max=2,
            notes="Enterprise productivity seat pricing comparison.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_06",
            query_text="Compare Meta Llama 3 open weights licensing terms with Mistral Large 2 commercial pricing and parameters.",
            category=BenchmarkCategory.MULTI_ENTITY,
            relevant_chunk_ids={"chunk_056": 3, "chunk_058": 3, "chunk_059": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=2,
            expected_subqueries_max=2,
            notes="Open weights vs proprietary API terms comparison across Meta and Mistral.",
        ),

        # Category 2: Multi-Faceted Queries (6 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_07",
            query_text="What are OpenAI's latest developments across enterprise pricing, SOC2 compliance, and supercomputing systems hiring?",
            category=BenchmarkCategory.MULTI_FACETED,
            relevant_chunk_ids={"chunk_018": 3, "chunk_021": 3, "chunk_022": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=3,
            expected_subqueries_max=3,
            notes="Decomposes into pricing, compliance, and hiring aspects for OpenAI.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_08",
            query_text="Detail Databricks' strategy across MosaicML team hiring, Tabular Iceberg acquisition, and serverless compute pricing.",
            category=BenchmarkCategory.MULTI_FACETED,
            relevant_chunk_ids={"chunk_012": 3, "chunk_014": 3, "chunk_015": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=3,
            expected_subqueries_max=3,
            notes="Decomposes into M&A, hiring, and pricing facets for Databricks.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_09",
            query_text="What is Anthropic's status regarding London alignment lab hiring, Message Batches discount, and Claude 3.5 coding benchmarks?",
            category=BenchmarkCategory.MULTI_FACETED,
            relevant_chunk_ids={"chunk_030": 3, "chunk_031": 3, "chunk_032": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=3,
            expected_subqueries_max=3,
            notes="Multi-faceted enquiry covering Anthropic R&D, batch API, and benchmark evals.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_10",
            query_text="What are Snowflake's enterprise capabilities regarding FedRAMP authorization, Arctic model release, and Data Clean Rooms?",
            category=BenchmarkCategory.MULTI_FACETED,
            relevant_chunk_ids={"chunk_001": 3, "chunk_006": 3, "chunk_007": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=3,
            expected_subqueries_max=3,
            notes="Covers security, LLM product, and data monetization clean rooms.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_11",
            query_text="Detail Google's Gemini strategy spanning London RL team expansion, 2M context window pricing, and Vertex AI provisioned throughput.",
            category=BenchmarkCategory.MULTI_FACETED,
            relevant_chunk_ids={"chunk_035": 3, "chunk_038": 3, "chunk_040": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=3,
            expected_subqueries_max=3,
            notes="Covers Google DeepMind research hiring, model pricing, and enterprise throughput SLA.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_12",
            query_text="Detail Qdrant's developments in quantization memory reduction, hybrid BM25 search, and Cloud SOC2 compliance.",
            category=BenchmarkCategory.MULTI_FACETED,
            relevant_chunk_ids={"chunk_044": 3, "chunk_046": 3, "chunk_048": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            expected_subqueries_min=3,
            expected_subqueries_max=3,
            notes="Covers vector database compression, lexical fusion, and security compliance.",
        ),

        # Category 3: Terminology Mismatch (5 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_13",
            query_text="Vector database low-cost cold storage techniques and NVMe disk payload paging.",
            category=BenchmarkCategory.TERMINOLOGY_MISMATCH,
            relevant_chunk_ids={"chunk_045": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Query uses generic terminology matching chunk_045 (disk-backed payload paging with scalar quantization).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_14",
            query_text="Asynchronous high-throughput non-realtime LLM request discounts and 24-hour turnaround pricing.",
            category=BenchmarkCategory.TERMINOLOGY_MISMATCH,
            relevant_chunk_ids={"chunk_020": 3, "chunk_031": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Matches OpenAI Batch API (chunk_020) and Anthropic Message Batches (chunk_031).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_15",
            query_text="Government sovereign cloud compliance and public sector classified analytics authorization.",
            category=BenchmarkCategory.TERMINOLOGY_MISMATCH,
            relevant_chunk_ids={"chunk_006": 3, "chunk_043": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Matches FedRAMP High authorizations for Snowflake (chunk_006) and Azure OpenAI (chunk_043).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_16",
            query_text="Speculative token generation acceleration mechanisms with small draft models.",
            category=BenchmarkCategory.TERMINOLOGY_MISMATCH,
            relevant_chunk_ids={"chunk_068": 3, "chunk_069": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Matches chunked prefill and draft model speculative decoding in vLLM.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_17",
            query_text="High-density multi-tenant vector memory compression and 1-bit binary representation.",
            category=BenchmarkCategory.TERMINOLOGY_MISMATCH,
            relevant_chunk_ids={"chunk_044": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Matches scalar quantization and 1-bit binary quantization in Qdrant.",
        ),

        # Category 4: Evidence-Gap / Corrective Retrieval (6 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_18",
            query_text="What is the full SLA and financial penalty credit refund schedule for OpenAI Enterprise uptime commitments?",
            category=BenchmarkCategory.EVIDENCE_GAP,
            relevant_chunk_ids={"chunk_021": 2, "chunk_025": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Initial query retrieves compliance overview (chunk_021); corrective hop retrieves explicit credit refund table (chunk_025).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_19",
            query_text="Databricks Unity Catalog governance capabilities and exact cross-cloud audit retention periods.",
            category=BenchmarkCategory.EVIDENCE_GAP,
            relevant_chunk_ids={"chunk_016": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Requires specific 365 days retention metric retrieved via targeted query.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_20",
            query_text="Anthropic Claude 3.5 Sonnet Tier 4 organization rate limits and token-per-minute quotas.",
            category=BenchmarkCategory.EVIDENCE_GAP,
            relevant_chunk_ids={"chunk_028": 2, "chunk_034": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="First hop retrieves general pricing; corrective hop retrieves 400k TPM rate limit details.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_21",
            query_text="Snowflake Cortex AI regional deployment availability and disaster recovery RTO/RPO metrics.",
            category=BenchmarkCategory.EVIDENCE_GAP,
            relevant_chunk_ids={"chunk_004": 2, "chunk_005": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="First hop retrieves Cortex pricing; corrective hop retrieves 4-hour RTO / 15-minute RPO SLA.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_22",
            query_text="Pinecone Serverless query latency claims and p99 degradation under 10k QPS load.",
            category=BenchmarkCategory.EVIDENCE_GAP,
            relevant_chunk_ids={"chunk_051": 2, "chunk_052": 3, "chunk_053": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Retrieves serverless architecture, vendor claim, and independent benchmark degradation.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_23",
            query_text="Qdrant hybrid search index construction memory overhead and sparse RAM multiplier.",
            category=BenchmarkCategory.EVIDENCE_GAP,
            relevant_chunk_ids={"chunk_046": 2, "chunk_047": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="First hop gets hybrid overview; corrective hop retrieves 1.35x RAM multiplier detail.",
        ),

        # Category 5: Cross-Document Multi-Hop (6 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_24",
            query_text="What was the financial deal size of Databricks' Tabular acquisition and what is the scheduled Unity Catalog integration date?",
            category=BenchmarkCategory.CROSS_DOCUMENT,
            relevant_chunk_ids={"chunk_012": 3, "chunk_013": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Doc A has $1.5B acquisition price, Doc B has Q4 2024 Unity Catalog integration roadmap.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_25",
            query_text="How did OpenAI's acquisition of Rockset impact its enterprise real-time search roadmap and indexing latency?",
            category=BenchmarkCategory.CROSS_DOCUMENT,
            relevant_chunk_ids={"chunk_023": 3, "chunk_024": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Doc A has Rockset M&A announcement, Doc B has resulting 18ms p95 latency reduction.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_26",
            query_text="Connect Anthropic's London alignment lab setup with their published mechanistic interpretability findings.",
            category=BenchmarkCategory.CROSS_DOCUMENT,
            relevant_chunk_ids={"chunk_032": 3, "chunk_033": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Doc A has London lab opening, Doc B has monosemantic feature extraction findings.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_27",
            query_text="What is Snowflake's Arctic inference throughput on NVIDIA H100 and how does that affect customer warehouse billing credits?",
            category=BenchmarkCategory.CROSS_DOCUMENT,
            relevant_chunk_ids={"chunk_004": 2, "chunk_008": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Doc A has 142 tokens/sec H100 benchmark, Doc B has 2.2 warehouse credit consumption rate.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_28",
            query_text="Trace xAI's Memphis supercluster power infrastructure to their target model parameter count for Grok-3.",
            category=BenchmarkCategory.CROSS_DOCUMENT,
            relevant_chunk_ids={"chunk_065": 3, "chunk_066": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Doc A has Memphis 150MW 100k GPU cluster, Doc B has 1.2T parameter target.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_29",
            query_text="What are the licensing conditions for deploying Meta's Llama 3 in commercial SaaS and what is the active monthly user threshold requiring special permission?",
            category=BenchmarkCategory.CROSS_DOCUMENT,
            relevant_chunk_ids={"chunk_055": 2, "chunk_056": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Doc A has Llama 3 models, Doc B has 700M monthly active user threshold clause.",
        ),

        # Category 6: Temporal / Version Change (5 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_30",
            query_text="How did Anthropic's pricing and coding benchmark scores change from Claude 3.0 Opus to Claude 3.5 Sonnet?",
            category=BenchmarkCategory.TEMPORAL_CHANGE,
            relevant_chunk_ids={"chunk_028": 3, "chunk_029": 3, "chunk_030": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Compares Claude 3.0 Opus ($15/$75, 84.9%) with Claude 3.5 Sonnet ($3/$15, 92.0%).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_31",
            query_text="What were the context window expansions for Google Gemini from v1.0 Ultra to v1.5 Pro?",
            category=BenchmarkCategory.TEMPORAL_CHANGE,
            relevant_chunk_ids={"chunk_035": 3, "chunk_036": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Compares Gemini 1.0 (32k context) with Gemini 1.5 Pro (2M context).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_32",
            query_text="How did OpenAI's flagship model pricing change between GPT-4 Turbo and GPT-4o?",
            category=BenchmarkCategory.TEMPORAL_CHANGE,
            relevant_chunk_ids={"chunk_018": 3, "chunk_019": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Compares GPT-4 Turbo ($10/$30) with GPT-4o ($5/$15).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_33",
            query_text="Track the evolution of vLLM throughput improvements between v0.3.0 and v0.4.2.",
            category=BenchmarkCategory.TEMPORAL_CHANGE,
            relevant_chunk_ids={"chunk_067": 3, "chunk_068": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Compares v0.3.0 baseline with v0.4.2 chunked prefill 2.8x speedup.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_34",
            query_text="Compare Snowflake Arctic v1.0 architecture with the Arctic v2 preview announcement.",
            category=BenchmarkCategory.TEMPORAL_CHANGE,
            relevant_chunk_ids={"chunk_001": 3, "chunk_003": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Compares Arctic v1 (480B MoE) with Arctic v2 (dense-MoE hybrid preview).",
        ),

        # Category 7: Contradictory Evidence (5 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_35",
            query_text="What is the reported p99 query latency for Pinecone Serverless under 10k QPS?",
            category=BenchmarkCategory.CONTRADICTORY,
            relevant_chunk_ids={"chunk_052": 3, "chunk_053": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Vendor whitepaper reports 12ms p99 vs independent benchmark reporting 45ms p99.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_36",
            query_text="What is the published token generation speed for Mistral Large 2 on 8x H100 nodes?",
            category=BenchmarkCategory.CONTRADICTORY,
            relevant_chunk_ids={"chunk_062": 3, "chunk_063": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Official release notes claim 85 tokens/sec vs third-party hosting report of 42 tokens/sec.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_37",
            query_text="What is OpenAI GPT-4o real-time voice latency?",
            category=BenchmarkCategory.CONTRADICTORY,
            relevant_chunk_ids={"chunk_026": 3, "chunk_027": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Marketing claim of 232ms vs independent API measurement of 650ms.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_38",
            query_text="What is Databricks Lakehouse SQL execution cost and speed compared to Snowflake on TPC-DS?",
            category=BenchmarkCategory.CONTRADICTORY,
            relevant_chunk_ids={"chunk_009": 3, "chunk_017": 3},
            expected_decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
            notes="Databricks study claims 2.5x lower cost vs Snowflake study claiming 1.8x faster and 35% cheaper.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_39",
            query_text="What is Qdrant's memory consumption per 1M 1536-dim vectors with scalar quantization?",
            category=BenchmarkCategory.CONTRADICTORY,
            relevant_chunk_ids={"chunk_049": 3, "chunk_050": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Official documentation states 2.4 GB vs community load test reporting 4.1 GB.",
        ),

        # Category 8: Adversarial / Negative Cases (6 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_40",
            query_text="What is the pricing and release date for Apple Intelligence on Android devices?",
            category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
            relevant_chunk_ids={},  # True negative: non-existent feature
            is_negative_case=True,
            distractor_chunk_ids=["chunk_070"],
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Adversarial: Non-existent feature; chunk_070 clarifies zero support for Android.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_41",
            query_text="How does Snowflake's agricultural crop yield prediction engine integrate with farm telemetry?",
            category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
            relevant_chunk_ids={},  # True negative: entity mismatch
            is_negative_case=True,
            distractor_chunk_ids=["chunk_071"],
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Adversarial: Keyword distractor matching agricultural snowflake refrigeration.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_42",
            query_text="What is the API pricing for Google Gemini models on AWS Bedrock?",
            category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
            relevant_chunk_ids={},  # True negative: non-existent deployment
            is_negative_case=True,
            distractor_chunk_ids=["chunk_074"],
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Adversarial: AWS Bedrock supports Anthropic/Meta but not Google Gemini (chunk_074).",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_43",
            query_text="What were the financial terms of OpenAI's acquisition of Databricks?",
            category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
            relevant_chunk_ids={},  # True negative: fictitious M&A
            is_negative_case=True,
            distractor_chunk_ids=["chunk_012", "chunk_023", "chunk_075"],
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Adversarial: Fictitious merger; distractors discuss Tabular and Rockset acquisitions.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_44",
            query_text="Compare the llama wool grading standards between Peruvian farms and Meta AI.",
            category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
            relevant_chunk_ids={},  # True negative: homonym ambiguity
            is_negative_case=True,
            distractor_chunk_ids=["chunk_055", "chunk_072"],
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Adversarial: Ambiguous homonym between biological llama farming and Meta Llama model.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_45",
            query_text="What is the GPU acceleration roadmap for SQLite vector search?",
            category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
            relevant_chunk_ids={},  # True negative: non-existent feature
            is_negative_case=True,
            distractor_chunk_ids=["chunk_073"],
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Adversarial: SQLite vector search has no GPU acceleration roadmap (chunk_073).",
        ),

        # Category 9: Direct Single-Hop Lookups (5 Queries)
        MultiHopEvaluationQuery(
            query_id="mh_query_46",
            query_text="What is the price per 1M input tokens for Google Gemini 1.5 Flash?",
            category=BenchmarkCategory.DIRECT_LOOKUP,
            relevant_chunk_ids={"chunk_037": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Direct single-hop lookup for Gemini Flash pricing.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_47",
            query_text="What is GitHub Copilot Enterprise monthly per-user subscription price?",
            category=BenchmarkCategory.DIRECT_LOOKUP,
            relevant_chunk_ids={"chunk_042": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Direct single-hop lookup for GitHub Copilot Enterprise pricing.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_48",
            query_text="What context window length was announced for Mistral NeMo?",
            category=BenchmarkCategory.DIRECT_LOOKUP,
            relevant_chunk_ids={"chunk_060": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Direct single-hop lookup for Mistral NeMo 128k context.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_49",
            query_text="Where is Mistral AI hiring distributed training engineers?",
            category=BenchmarkCategory.DIRECT_LOOKUP,
            relevant_chunk_ids={"chunk_061": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Direct single-hop lookup for Mistral Paris engineering openings.",
        ),
        MultiHopEvaluationQuery(
            query_id="mh_query_50",
            query_text="What parameter size was open-sourced for xAI Grok-1?",
            category=BenchmarkCategory.DIRECT_LOOKUP,
            relevant_chunk_ids={"chunk_064": 3},
            expected_decision=RetrievalDecision.DIRECT_RETRIEVAL,
            notes="Direct single-hop lookup for Grok-1 314B MoE parameters.",
        ),
    ]

    return MultiHopEvaluationDataset(
        dataset_id="ci_multihop_adversarial_benchmark_v1",
        corpus=corpus,
        queries=queries,
    )
