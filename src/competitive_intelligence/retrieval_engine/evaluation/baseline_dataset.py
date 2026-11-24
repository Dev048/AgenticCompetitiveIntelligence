from datetime import datetime, timezone
import uuid

from competitive_intelligence.knowledge_pipeline.models import ChunkMetadata, DocumentChunk
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash
from competitive_intelligence.retrieval_engine.evaluation.models import EvaluationDataset, EvaluationQuery


def _create_chunk(content: str, source_category: str, company_id: str = "unknown") -> DocumentChunk:
    cid = str(uuid.uuid4())
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    meta = ChunkMetadata(
        source_id="src_baseline",
        document_id=doc_id,
        version_id="v1",
        company_id=company_id,
        workspace_id="ws_eval",
        source_url="https://example.com/baseline",
        source_category=source_category,
        collected_at=datetime.now(timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=0,
        total_chunks=1,
    )
    return DocumentChunk(
        chunk_id=cid,
        document_id=doc_id,
        version_id="v1",
        content=content,
        chunk_index=0,
        metadata=meta,
        content_hash=compute_content_hash(content),
        created_at=datetime.now(timezone.utc),
    )


def get_baseline_dataset() -> EvaluationDataset:
    """Generate the baseline evaluation dataset with 50-75 chunks and 20-30 queries."""
    
    # Define raw chunk data
    raw_chunks = [
        # Pricing
        ("OpenAI announced new GPT-4o pricing at $5 per 1M input tokens and $15 per 1M output tokens, a 50% drop.", "pricing", "openai"),
        ("Anthropic Claude 3.5 Sonnet pricing is $3 per 1M input tokens and $15 per 1M output tokens.", "pricing", "anthropic"),
        ("Google Gemini 1.5 Pro is priced at $3.50 per 1M input tokens (up to 128k context).", "pricing", "google"),
        ("OpenAI Batch API offers a 50% discount for asynchronous workloads.", "pricing", "openai"),
        ("Anthropic introduces Message Batches with a 50% discount for non-urgent tasks.", "pricing", "anthropic"),
        ("Cohere Command R+ pricing focuses on enterprise volume discounts starting at $10k/month.", "pricing", "cohere"),
        ("Mistral Large 2 costs $3 per 1M input and $9 per 1M output tokens via La Plateforme.", "pricing", "mistral"),
        ("Together AI reduces Llama 3 70B inference pricing to $0.88 per 1M tokens.", "pricing", "together"),
        ("DeepSeek-Coder-V2 API pricing remains highly disruptive at $0.14 per 1M input tokens.", "pricing", "deepseek"),
        ("OpenAI Enterprise plan custom pricing requires contacting sales; estimated at $60/user/month.", "pricing", "openai"),
        ("GitHub Copilot Enterprise is priced at $39 per user per month.", "pricing", "microsoft"),
        ("AWS Bedrock provisioned throughput pricing for Claude 3 requires a 1-month minimum commitment.", "pricing", "amazon"),

        # Releases / News
        ("OpenAI releases GPT-4o with native multimodal capabilities including real-time audio and vision.", "news", "openai"),
        ("Anthropic launches Claude 3.5 Sonnet, outperforming Opus on standard coding benchmarks.", "news", "anthropic"),
        ("Google announces Gemini 1.5 Pro with a massive 2 million token context window.", "news", "google"),
        ("Mistral AI unveils Mistral Large 2, a 123B parameter model competitive with GPT-4 class models.", "news", "mistral"),
        ("Meta releases Llama 3 with 8B and 70B parameter versions, achieving state-of-the-art open weights performance.", "news", "meta"),
        ("xAI open-sources Grok-1, a 314B parameter Mixture-of-Experts model.", "news", "xai"),
        ("Apple Intelligence announced with on-device small language models and Private Cloud Compute.", "news", "apple"),
        ("OpenAI acquires Rockset to enhance enterprise search and retrieval capabilities.", "news", "openai"),
        ("Databricks acquires Tabular for $1-2 billion to unify Apache Iceberg and Linux Foundation Delta Lake.", "news", "databricks"),
        ("Snowflake announces Arctic, a highly efficient open enterprise LLM focused on SQL generation.", "news", "snowflake"),

        # GitHub / Tech
        ("vLLM v0.4.0 released with chunked prefill support and speculative decoding enhancements.", "github", "vllm"),
        ("Transformers v4.40.0 adds native support for Llama 3 and Mistral architectures.", "github", "huggingface"),
        ("LangChain v0.2 introduces a more modular architecture separating core from partner packages.", "github", "langchain"),
        ("LlamaIndex v0.10 moves integrations into separate packages to reduce dependency bloat.", "github", "llamaindex"),
        ("Qdrant v1.9 releases with native support for sparse vectors (BM25) and multitenancy improvements.", "github", "qdrant"),
        ("Milvus 2.4 introduces GPU-accelerated indexing with CAGRA and refined RBAC.", "github", "zilliz"),
        ("TensorRT-LLM v0.9 optimizes Hopper architecture performance and adds FP8 quantization support.", "github", "nvidia"),
        ("Ollama v0.1.30 adds support for Gemma 2 and improved concurrent request handling.", "github", "ollama"),
        ("HuggingFace TGI v2.0 introduces prefix caching and Flash Attention 2 integration.", "github", "huggingface"),
        ("DSPy v2.4 optimizes teleprompters for few-shot pipeline compilation.", "github", "stanford"),

        # Careers / Team
        ("OpenAI is aggressively hiring Systems Software Engineers for their supercomputing team.", "careers", "openai"),
        ("Anthropic posts openings for Alignment Researchers focused on Mechanistic Interpretability.", "careers", "anthropic"),
        ("Google DeepMind expands the Gemini reinforcement learning team in London.", "careers", "google"),
        ("Mistral AI is looking for Distributed Training Engineers based in Paris.", "careers", "mistral"),
        ("xAI is hiring Data Engineers in the Bay Area to scale their Memphis supercluster pipeline.", "careers", "xai"),
        ("Cohere opens a new office in New York and is hiring Enterprise Sales Directors.", "careers", "cohere"),
        ("Scale AI is hiring prompt engineers with advanced degrees in mathematics and law.", "careers", "scale"),
        ("Databricks is expanding its MosaicML team, hiring compiler engineers for LLM training.", "careers", "databricks"),

        # Community / Reddit / Blog
        ("Reddit users note that Claude 3.5 Sonnet's Artifacts feature is a game-changer for UI prototyping.", "reddit", "anthropic"),
        ("Developers on Twitter complain about GPT-4o API rate limits during the initial rollout.", "reddit", "openai"),
        ("Hacker News thread discusses the impressive reasoning capabilities of Gemini 1.5 Pro with large codebases.", "reddit", "google"),
        ("A popular blog post highlights how Llama 3 70B can replace GPT-4 for 80% of enterprise RAG pipelines.", "blog", "meta"),
        ("Enterprise users are frustrated by the lack of clear SOC2 compliance documentation for new AI startups.", "reddit", "unknown"),
        ("Benchmark blog post shows vLLM achieves 2x throughput over TGI on Llama 3 inference.", "blog", "vllm"),
        ("Community consensus is that Qdrant is currently the most developer-friendly Rust vector database.", "reddit", "qdrant"),
        ("Users report that Mistral Large 2 excels at multi-lingual tasks compared to equivalent Llama models.", "reddit", "mistral"),
        ("A deep dive article explains how Mixture of Experts (MoE) reduces inference costs for Grok-1.", "blog", "xai"),
        ("Reddit r/LocalLLaMA users highly recommend the BGE-M3 embedding model for multilingual semantic search.", "reddit", "baai"),
        
        # Filler chunks to hit 50+
        ("The competitive intelligence market is projected to grow to $4 billion by 2030.", "news", "unknown"),
        ("Understanding the strategic moves of cloud hyperscalers is critical for AI startups.", "blog", "unknown"),
        ("Venture capital funding for generative AI infrastructure reached $10B in Q2.", "news", "unknown"),
        ("AI agents are transitioning from autonomous execution to bounded, graph-based workflows.", "blog", "unknown"),
        ("Retrieval-Augmented Generation (RAG) remains the standard for reducing LLM hallucinations.", "blog", "unknown"),
        ("Agentic RAG uses LLMs to route queries, plan searches, and evaluate retrieved evidence.", "blog", "unknown"),
        ("Reciprocal Rank Fusion (RRF) is a popular technique for combining dense and lexical search results.", "blog", "unknown"),
        ("Cross-encoder reranking provides higher accuracy but introduces significant latency.", "blog", "unknown"),
        ("Embedding models convert text into high-dimensional vectors for semantic search.", "blog", "unknown"),
        ("BM25 is a strong baseline for keyword-based retrieval, especially for exact matches.", "blog", "unknown"),
    ]

    corpus = []
    for content, cat, comp in raw_chunks:
        corpus.append(_create_chunk(content, cat, comp))
        
    # Helper to find chunk ID by a substring of its content
    def get_id(substring: str) -> str:
        for chunk in corpus:
            if substring in chunk.content:
                return chunk.chunk_id
        raise ValueError(f"Substring '{substring}' not found in corpus.")

    queries = [
        EvaluationQuery(
            query_id="q1",
            query_text="What is the pricing for OpenAI's GPT-4o API?",
            relevant_chunk_ids={
                get_id("GPT-4o pricing at $5 per 1M input"): 3,
                get_id("Batch API offers a 50% discount"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q2",
            query_text="Which model offers a 2 million token context window?",
            relevant_chunk_ids={
                get_id("Gemini 1.5 Pro with a massive 2 million"): 3,
                get_id("Gemini 1.5 Pro is priced at"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q3",
            query_text="Anthropic Claude 3.5 Sonnet artifacts and UI generation",
            relevant_chunk_ids={
                get_id("Artifacts feature is a game-changer for UI"): 3,
                get_id("Claude 3.5 Sonnet, outperforming Opus"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q4",
            query_text="Who is hiring mechanistic interpretability researchers?",
            relevant_chunk_ids={
                get_id("Alignment Researchers focused on Mechanistic Interpretability"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q5",
            query_text="vLLM chunked prefill support version",
            relevant_chunk_ids={
                get_id("vLLM v0.4.0 released with chunked prefill"): 3,
                get_id("vLLM achieves 2x throughput over TGI"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q6",
            query_text="BM25 and sparse vector support in Qdrant",
            relevant_chunk_ids={
                get_id("Qdrant v1.9 releases with native support for sparse vectors"): 3,
                get_id("Qdrant is currently the most developer-friendly"): 1,
                get_id("BM25 is a strong baseline"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q7",
            query_text="What is Mistral Large 2 pricing?",
            relevant_chunk_ids={
                get_id("Mistral Large 2 costs $3 per 1M input"): 3,
                get_id("Mistral AI unveils Mistral Large 2"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q8",
            query_text="OpenAI acquisitions for enterprise search",
            relevant_chunk_ids={
                get_id("OpenAI acquires Rockset to enhance enterprise search"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q9",
            query_text="Llama 3 70B replacing GPT-4 in enterprise RAG",
            relevant_chunk_ids={
                get_id("Llama 3 70B can replace GPT-4 for 80%"): 3,
                get_id("Llama 3 with 8B and 70B parameter versions"): 2,
            }
        ),
        EvaluationQuery(
            query_id="q10",
            query_text="xAI Grok-1 parameter count and architecture",
            relevant_chunk_ids={
                get_id("Grok-1, a 314B parameter Mixture-of-Experts model"): 3,
                get_id("Mixture of Experts (MoE) reduces inference costs for Grok-1"): 2,
            }
        ),
        EvaluationQuery(
            query_id="q11",
            query_text="DeepSeek Coder API cost",
            relevant_chunk_ids={
                get_id("DeepSeek-Coder-V2 API pricing remains highly disruptive"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q12",
            query_text="Apple intelligence and private cloud",
            relevant_chunk_ids={
                get_id("Apple Intelligence announced with on-device"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q13",
            query_text="Databricks acquiring Tabular and MosaicML",
            relevant_chunk_ids={
                get_id("Databricks acquires Tabular for $1-2 billion"): 3,
                get_id("expanding its MosaicML team"): 2,
            }
        ),
        EvaluationQuery(
            query_id="q14",
            query_text="GitHub Copilot Enterprise pricing per month",
            relevant_chunk_ids={
                get_id("GitHub Copilot Enterprise is priced at $39"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q15",
            query_text="HuggingFace TGI v2.0 features Flash Attention",
            relevant_chunk_ids={
                get_id("HuggingFace TGI v2.0 introduces prefix caching and Flash Attention 2"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q16",
            query_text="BGE-M3 multilingual embedding model reviews",
            relevant_chunk_ids={
                get_id("BGE-M3 embedding model for multilingual semantic search"): 3,
                get_id("Embedding models convert text into high-dimensional vectors"): 1,
            }
        ),
        EvaluationQuery(
            query_id="q17",
            query_text="What is Reciprocal Rank Fusion used for?",
            relevant_chunk_ids={
                get_id("Reciprocal Rank Fusion (RRF) is a popular technique for"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q18",
            query_text="Does Snowflake have an open enterprise LLM?",
            relevant_chunk_ids={
                get_id("Snowflake announces Arctic, a highly efficient open enterprise LLM"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q19",
            query_text="Mistral distributed training engineering jobs Paris",
            relevant_chunk_ids={
                get_id("Mistral AI is looking for Distributed Training Engineers based in Paris"): 3,
            }
        ),
        EvaluationQuery(
            query_id="q20",
            query_text="Batch inference discount across major providers",
            relevant_chunk_ids={
                get_id("Batch API offers a 50% discount"): 3,
                get_id("Anthropic introduces Message Batches with a 50% discount"): 3,
            }
        ),
    ]

    return EvaluationDataset(
        dataset_id="baseline_ci_v1",
        queries=queries,
        corpus=corpus
    )
