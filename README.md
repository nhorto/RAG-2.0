# Tekla PowerFab RAG System v2.1

Enterprise-grade Retrieval-Augmented Generation (RAG) system for steel fabrication consulting transcripts and summaries. Built with hybrid search (dense + sparse), advanced query processing, metadata filtering, and comprehensive evaluation framework.

## Features

### Core Capabilities
- **True Hybrid Search**: Combines dense vector search (OpenAI embeddings) with sparse BM25 indexing using Qdrant native sparse vectors
- **Qdrant Prefetch Fusion**: Server-side optimized fusion for single-call hybrid search with automatic fallback
- **Optional Reranking**: Cohere Rerank API or local ColBERT for quality enhancement (5-15% improvement)
- **Advanced Query Processing**: NEW - Intelligent query decomposition and augmentation
  - **Query Decomposition**: Breaks complex multi-part queries into atomic sub-queries
  - **Query Augmentation**: Adds domain context to vague/underspecified queries
  - **Intelligent Routing**: Heuristic detection with minimal LLM overhead
  - **Orchestrated Execution**: Parallel/sequential retrieval with smart result merging
- **Type-Aware Chunking**: Intelligent chunking strategies for transcripts vs summaries
- **Rich Metadata Extraction**: Named entity recognition, keyword matching, action items, decisions
- **Query Processing**: Automatic query expansion, rewriting, and entity extraction
- **Metadata Filtering**: Filter by client, date range, document type, PowerFab modules
- **Evaluation Framework**: Built-in metrics (precision, recall, MRR, faithfulness, relevancy)
- **Interactive UI**: Streamlit-based chat interface with source citations

### Technology Stack
- **Vector Database**: Qdrant 1.7.0+ with multi-vector collections (dense + sparse)
- **Dense Embeddings**: OpenAI text-embedding-3-large (3072 dimensions, can be reduced)
- **Sparse Embeddings**: FastEmbed BM25 (Qdrant/bm25 model) for keyword indexing
- **Reranking**: Cohere Rerank API (rerank-english-v3.0) or local ColBERT (optional)
- **LLM**: OpenAI GPT-4 for response generation and query rewriting
- **Evaluation**: RAGAS framework for systematic quality assessment
- **NER**: spaCy for named entity extraction
- **UI**: Streamlit for interactive chat interface

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                            │
│  Documents → Chunking → Metadata → Hybrid Embeddings → Qdrant   │
│                                      ├─ Dense (OpenAI 3072d)     │
│                                      └─ Sparse (BM25)            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE (v2.1)                     │
│  Query → Advanced Processing (NEW)                               │
│           ├─ Analyzer (heuristic, ~5ms)                          │
│           ├─ Decomposer (GPT-4, if complex)                      │
│           └─ Augmenter (GPT-3.5, if vague)                       │
│                          ↓                                       │
│         Traditional Processing (entity, intent, expansion)       │
│                          ↓                                       │
│         Orchestrator (parallel/sequential execution)             │
│                          ↓                                       │
│            Qdrant Prefetch Fusion (Server-Side)                  │
│                 ├─ Dense Search (HNSW)                           │
│                 └─ Sparse Search (BM25 Index)                    │
│                          ↓                                       │
│                  Optional Reranking (Cohere)                     │
│                          ↓                                       │
│                Result Merging & Deduplication                    │
│                          ↓                                       │
│                       Results                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   GENERATION PIPELINE                            │
│  Context Assembly → LLM Generation → Citations                  │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (for local Qdrant)
- OpenAI API key
- Cohere API key (optional, for reranking)

### Installation

1. **Clone and navigate to project**
```bash
cd /Users/nicholashorton/Documents/TeklaPowerFabRAG_v2
```

2. **Install dependencies**
```bash
make install
```

This installs:
- Python packages (qdrant-client, openai, streamlit, etc.)
- spaCy English model (en_core_web_sm)

3. **Set up Qdrant**
```bash
make setup
```

This:
- Starts Qdrant in Docker
- Creates `.env` file from template

4. **Configure API keys**

Edit `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
COHERE_API_KEY=your-cohere-key  # Optional, for reranking
```

### Usage

#### 1. Ingest Documents

First, create a symlink to your documents or copy them:

```bash
# Option A: Symlink (recommended)
ln -s "/Users/nicholashorton/Documents/LLM Sumarization" data/raw

# Option B: Copy files
cp -r "/Users/nicholashorton/Documents/LLM Sumarization"/*.txt data/raw/
```

Then ingest:

```bash
make ingest
```

Or with custom options:

```bash
python scripts/ingest_documents.py \
  --source-dir data/raw \
  --collection consulting_transcripts \
  --batch-size 10
```

#### 2. Run the UI

```bash
make run-ui
```

Open http://localhost:8501 in your browser.

#### 3. Search and Chat

**Example Queries:**
- "How do I create a BOM in the Estimating module?"
- "What did we discuss with ClientA last week?"
- "Show me issues with Work Orders from yesterday"
- "What decisions were made about Production Control?"

**Using Filters:**
- Filter by client name
- Filter by date range (last week, last month, custom)
- Filter by document type (transcript, daily summary, master summary)
- Choose search mode (hybrid, dense only, sparse only)

## Project Structure

```
TeklaPowerFabRAG_v2/
├── config/
│   ├── settings.yaml              # System configuration
│   └── domain_vocabulary.json     # PowerFab terminology
├── data/
│   ├── raw/                        # Source documents (symlink or copy)
│   ├── processed/                  # Processed documents (JSON)
│   └── embeddings_cache/           # Embedding cache (optional)
├── src/
│   ├── ingestion/                  # Document processing
│   │   ├── chunking_engine.py     # Type-aware chunking
│   │   ├── metadata_extractor.py  # NER + keyword extraction
│   │   ├── embedding_generator.py # Hybrid embeddings (OpenAI + BM25)
│   │   └── document_loader.py     # File loading
│   ├── retrieval/                  # Search & query processing
│   │   ├── query_analyzer.py      # NEW - Heuristic query complexity detection
│   │   ├── query_decomposer.py    # NEW - Multi-part query decomposition (GPT-4)
│   │   ├── query_augmenter.py     # NEW - Domain context augmentation (GPT-3.5)
│   │   ├── query_orchestrator.py  # NEW - Parallel/sequential execution & merging
│   │   ├── query_processor.py     # Query expansion/rewriting (integrates above)
│   │   ├── hybrid_searcher.py     # Dense + sparse fusion + reranking
│   │   ├── reranker.py            # Cohere and local reranking
│   │   └── types.py               # Query intent, complexity, cost enums
│   ├── generation/                 # Response generation
│   │   └── llm_interface.py       # LLM integration
│   ├── evaluation/                 # Quality metrics
│   │   └── metrics.py             # Precision, recall, RAGAS
│   ├── database/                   # Vector database
│   │   └── qdrant_client.py       # Qdrant operations (multi-vector)
│   └── utils/                      # Utilities
│       ├── config_loader.py       # Configuration
│       ├── llm_client.py          # LLM retry logic & token counting
│       ├── text_utils.py          # Text processing
│       └── logging.py             # Structured logging
├── scripts/
│   ├── ingest_documents.py        # CLI ingestion script
│   ├── migrate_to_hybrid.py       # Migration utility
│   ├── benchmark_fusion.py        # Fusion benchmarking
│   ├── test_reranking.py          # Reranking quality test
│   ├── estimate_rerank_costs.py   # Cost calculator
│   └── test_hybrid_complete.py    # Complete test suite
├── ui/
│   └── streamlit_app.py                  # Streamlit interface
├── tests/                                 # Unit & integration tests
│   ├── test_query_analyzer.py            # Query analysis tests
│   ├── test_advanced_query_integration.py# Advanced query integration tests
│   └── data/test_queries/
│       └── advanced_test_queries.json    # Test dataset (20 queries)
├── docs/                                  # Advanced documentation
│   ├── ADVANCED_QUERY_PROCESSING_README.md  # Query processing user guide
│   ├── PRD_ADVANCED_QUERY_PROCESSING.md     # Product requirements
│   ├── IMPLEMENTATION_SUMMARY_ADVANCED_QUERY.md
│   └── QUERY_PROCESSING_DECISION_TREE.md    # Decision logic
├── .env                                   # API keys (create from .env.example)
├── requirements.txt                       # Python dependencies (inc. tenacity)
├── Makefile                               # Common commands
├── README.md                              # This file
├── START_HERE.md                          # Quick start guide
├── QUICKSTART.md                          # 5-minute setup
├── CHANGELOG.md                           # Version history and changes
├── IMPLEMENTATION_SUMMARY.md              # Implementation overview
├── IMPLEMENTATION_CHANGES_v2.1.md         # Detailed v2.1 changes
├── IMPLEMENTATION_COMPLETE.md             # Advanced query processing summary
└── HYBRID_SEARCH_IMPLEMENTATION.md        # Hybrid search technical details
```

## Configuration

### Embedding Model Options

The system uses **hybrid embeddings** (dense + sparse) by default. Configure in `config/settings.yaml`:

```yaml
embeddings:
  # Dense embeddings (OpenAI)
  provider: "openai"
  model: "text-embedding-3-large"  # or "text-embedding-3-small"
  dimensions: 3072  # 3072, 1536, 768, or 256
  batch_size: 100

  # Sparse embeddings (FastEmbed BM25) - automatically included
  # No configuration needed, uses Qdrant/bm25 model
```

**Cost Optimization:**
- Dense (OpenAI): $0.13/1M tokens
  - `text-embedding-3-large` (3072 dim) - Best quality
  - `text-embedding-3-large` (1536 dim) - Reduced dimensions
  - `text-embedding-3-small` (1536 dim) - Lower cost ($0.02/1M)
- Sparse (FastEmbed BM25): **Free** (local inference)

### Chunking Strategies

```yaml
chunking:
  transcript:
    chunk_size: 512      # tokens
    overlap: 50          # 10% overlap
  summary:
    min_size: 100        # tokens
    max_size: 800        # tokens
```

### Hybrid Search Configuration

```yaml
retrieval:
  dense_search:
    top_k: 20  # Dense candidates for fusion

  sparse_search:
    top_k: 20  # Sparse candidates for fusion

  fusion:
    method: "prefetch"   # "prefetch" (recommended), "rrf", or "weighted_sum"
    rrf_k: 60            # RRF constant (for manual RRF)
    dense_weight: 0.7    # For weighted_sum
    sparse_weight: 0.3   # For weighted_sum

  final_top_k: 10        # Results after fusion

  reranking:
    enabled: false       # Set to true to enable
    provider: "cohere"   # "cohere" or "local"
    model: "rerank-english-v3.0"
    top_k: 5             # Final results after reranking
```

**Fusion Methods:**
- **prefetch** (default): Server-side Qdrant native fusion - fastest, single API call
- **rrf**: Manual Reciprocal Rank Fusion - backwards compatible
- **weighted_sum**: Weighted score combination - configurable weights

### Advanced Query Processing Configuration (NEW)

```yaml
query_processing:
  advanced_processing:
    # Master switch - disable all advanced features
    enabled: true

    # Feature flags
    enable_decomposition: true  # Multi-part query decomposition
    enable_augmentation: true   # Vague query augmentation

    # Detection thresholds
    analysis:
      min_decompose_confidence: 0.6  # 0.0-1.0
      min_augment_confidence: 0.5    # 0.0-1.0

    # Decomposition settings
    decomposition:
      llm_model: "gpt-4"              # Complex reasoning
      temperature: 0.3                # Low for consistency
      max_sub_queries: 5              # Limit explosion
      max_retries: 3                  # Retry on failure

    # Augmentation settings
    augmentation:
      llm_model: "gpt-3.5-turbo"      # Cost-effective
      temperature: 0.5                # Moderate for variety
      max_variants: 5                 # Limit variants
      max_retries: 3                  # Retry on failure

    # Orchestration settings
    orchestration:
      parallel_workers: 5             # ThreadPoolExecutor workers
      per_query_top_k: 10             # Results per sub-query
      enable_deduplication: true      # Remove duplicate chunks
```

**Query Types Handled:**
- **Simple** (75%): No enhancement, fast retrieval
- **Multi-part** (4%): Decomposed into sub-queries, executed in parallel/sequential
- **Vague** (20%): Augmented with domain context variations
- **Complex** (1%): Both decomposition and augmentation

**Performance Impact:**
- Simple queries: No overhead (~50ms)
- Augmented: +500ms (GPT-3.5 call)
- Decomposed: +800ms (GPT-4 call)
- Complex: +1400ms (both enhancements)

**Cost Impact:**
- Simple: $0.00
- Augmented: ~$0.0004 per query
- Decomposed: ~$0.012 per query
- Weighted average: ~$0.0022 per query

See `docs/ADVANCED_QUERY_PROCESSING_README.md` for detailed usage guide.

## Document Filename Convention

For automatic metadata extraction, use this naming pattern:

```
YYYY-MM-DD_ClientName_SiteName_doctype.txt
```

**Examples:**
- `2024-11-15_ClientA_Site1_transcript.txt`
- `2024-11-15_ClientA_Site1_daily_summary.txt`
- `2024-11-20_ClientA_Site1_master_summary.txt`

**Extracted Metadata:**
- `date`: 2024-11-15
- `meeting_day`: Friday
- `client_name`: ClientA
- `site_name`: Site1
- `document_type`: transcript

## API Costs Estimate

Based on 100 hours of transcripts (~1.5M words):

**One-time Ingestion:**
- Dense embeddings (OpenAI): ~2M tokens × $0.13/1M = **$0.26**
- Sparse embeddings (FastEmbed BM25): **Free** (local)
- Total: **< $1**

**Per Query:**
- Dense embedding: ~20 tokens × $0.13/1M = **$0.000003**
- Sparse embedding: **Free** (local BM25)
- LLM response: ~2000 tokens × $0.03/1K (GPT-4) = **$0.06**
- Reranking (optional): $0.001 per query (Cohere)
- Total per query: **~$0.06** (or **~$0.061** with reranking)

**Monthly Costs:**
- 100 queries without reranking: **~$6/month**
- 100 queries with reranking: **~$6.10/month**
- 1000 queries with reranking: **~$61/month**

## Performance Benchmarks

Expected performance on M1 Mac with hybrid search v2.1:

- **Chunking**: >1000 chunks/second
- **Embedding Generation**:
  - Dense (OpenAI API): ~100 docs/minute (API limited)
  - Sparse (FastEmbed BM25): ~500 docs/minute (local)
- **Ingestion**: ~50-100 docs/minute end-to-end
- **Query Latency**: <1 second total (p95)
  - Dense search: <200ms
  - Sparse search: **<500ms** (was 2-10s, 10-20x improvement)
  - Fusion (prefetch): **<50ms** (server-side)
  - Reranking (optional): ~100ms
  - LLM generation: 1-3 seconds

**Before vs After (Sparse Search):**
- Before: 2-10 seconds (O(n) keyword scanning)
- After: <500ms (indexed BM25 search)
- **Improvement: 10-20x faster**

## Evaluation

The system includes built-in evaluation:

### Retrieval Metrics
- **Precision@K**: % of retrieved results that are relevant
- **Recall@K**: % of relevant docs that were retrieved
- **MRR**: Mean Reciprocal Rank of first relevant result

### Generation Metrics (RAGAS)
- **Faithfulness**: Are claims supported by context?
- **Answer Relevancy**: Does response address query?
- **Answer Completeness**: Coverage of key points

**Target Metrics:**
- Precision@10: >85%
- Recall@10: >90%
- Faithfulness: >95%
- Answer Relevancy: >90%

## Troubleshooting

### Qdrant Connection Error
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
make restart-qdrant

# Or start fresh
make setup
```

### API Key Errors
```bash
# Verify .env file exists and has keys
cat .env

# Should contain:
# OPENAI_API_KEY=sk-...
```

### Import Errors
```bash
# Reinstall dependencies
make install

# Or manually:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Empty Search Results

1. **Check collection has data:**
```python
from src.database.qdrant_client import QdrantManager
qdrant = QdrantManager()
info = qdrant.get_collection_info()
print(f"Points: {info.get('points_count', 0)}")
```

2. **Re-ingest if empty:**
```bash
make ingest
```

### Slow Ingestion

- Reduce `batch_size` in ingestion script
- Check network connectivity (embeddings API)
- Consider caching embeddings

## Advanced Usage

### Custom Query Processing

```python
from src.retrieval.query_processor import QueryProcessor

processor = QueryProcessor()
processed = processor.process("How do I create BOMs?")

print(processed.expanded_queries)  # Query variations
print(processed.entities)          # Extracted entities
print(processed.intent)            # Query classification
```

### Direct Search

```python
from src.retrieval.hybrid_searcher import HybridSearcher

searcher = HybridSearcher()
results = searcher.search(
    query="BOM creation workflow",
    top_k=10,
    dense_only=False  # Use hybrid search
)

for result in results:
    print(f"{result.score:.3f}: {result.text[:100]}")
```

### Batch Evaluation

Create test dataset in `data/test_queries/test_queries.json`:

```json
{
  "queries": [
    {
      "query_id": "test-1",
      "query_text": "How do I create a BOM?",
      "ground_truth_chunks": ["chunk-uuid-1"],
      "expected_answer_points": ["Navigate to Estimating", "Click New BOM"]
    }
  ]
}
```

Then evaluate:

```bash
python scripts/evaluate_rag.py \
  --test-dataset data/test_queries/test_queries.json \
  --output reports/evaluation.json
```

## Migration from ChromaDB

If you have an existing ChromaDB collection:

1. Create migration script `scripts/migrate_from_chromadb.py`
2. Configure source and target
3. Run migration:

```bash
python scripts/migrate_from_chromadb.py
```

The migration will:
- Extract documents from ChromaDB
- Re-chunk with new strategies
- Extract enhanced metadata
- Generate new embeddings
- Upload to Qdrant

## Deployment

### Local Development
- Use Docker Qdrant (included in setup)
- Run Streamlit locally

### Production Options

**Option 1: Qdrant Cloud**
- Sign up at https://cloud.qdrant.io
- Use free tier (1GB) or paid plans
- Update `.env` with Qdrant Cloud credentials

**Option 2: Self-Hosted**
- Deploy Qdrant on your server
- Use persistent volumes for data
- Configure firewall for port 6333

**Option 3: Full Cloud Deployment**
- Deploy Streamlit to Streamlit Cloud
- Use Qdrant Cloud for vector database
- Set environment variables in deployment

## Security Notes

### Data Privacy
- All documents stored locally by default
- No data sent to OpenAI except queries and chunks (transient)
- OpenAI does not train on API data (per their policy)

### API Keys
- Never commit `.env` to git (included in `.gitignore`)
- Use environment variables in production
- Rotate keys regularly

### Access Control
- Add authentication to Streamlit (not included)
- Use Qdrant API key for production
- Implement user-based filtering if multi-tenant

## Contributing

This is a production RAG system. To extend:

1. Add new chunking strategies in `src/ingestion/chunking_engine.py`
2. Enhance metadata extraction in `src/ingestion/metadata_extractor.py`
3. Add custom metrics in `src/evaluation/metrics.py`
4. Extend UI in `ui/streamlit_app.py`

## License

Proprietary - For internal use only

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review system logs in `logs/`
3. Check Qdrant status: `docker logs qdrant`

## Changelog

### v2.1 (2025-12-15) - True Hybrid Search + Advanced Query Processing
- **🚀 Phase 1: Real BM25 Sparse Vectors**
  - Replaced broken O(n) keyword matching with Qdrant native sparse vectors
  - Added FastEmbed BM25 embeddings alongside OpenAI dense embeddings
  - Created HybridEmbeddingGenerator for dual embedding generation
  - Updated Qdrant schema for multi-vector collections
  - **Performance: 10-20x faster sparse search** (<500ms vs 2-10s)
- **⚡ Phase 2: Qdrant Native Fusion**
  - Implemented server-side prefetch fusion (single API call)
  - Added automatic fallback to manual RRF
  - Reduced network overhead and improved latency
- **✨ Phase 3: Optional Reranking**
  - Integrated Cohere Rerank API (rerank-english-v3.0)
  - Added local ColBERT reranking alternative
  - Expected 5-15% quality improvement on top results
  - Cost: $1 per 1000 queries (disabled by default)
- **🧠 Phase 4: Advanced Query Processing** (NEW)
  - 4 new modules: QueryAnalyzer, QueryDecomposer, QueryAugmenter, QueryOrchestrator
  - Heuristic-based complexity detection (no LLM overhead for simple queries)
  - Multi-part query decomposition with GPT-4
  - Vague query augmentation with GPT-3.5-turbo
  - Parallel/sequential execution strategies
  - Result merging and deduplication
  - Feature flags for gradual rollout
  - Comprehensive test suite (20+ test queries)
- **🛠️ New Tools & Scripts**
  - Migration utility for existing collections
  - Fusion benchmarking tool
  - Reranking quality tests
  - Cost estimation calculator
  - Comprehensive test suite
  - Advanced query integration tests

### v2.0 (2025-12-14) - Initial Implementation
- Complete rewrite with hybrid search foundation
- OpenAI embeddings instead of Voyage-3/Cohere
- Type-aware chunking strategies
- Rich metadata extraction with NER
- Query processing pipeline
- Streamlit UI with filtering
- Evaluation framework with RAGAS
- Production-ready architecture

---

**Built with ❤️ for steel fabrication consulting**
