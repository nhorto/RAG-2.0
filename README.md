# Tekla PowerFab RAG System v2.0

Enterprise-grade Retrieval-Augmented Generation (RAG) system for steel fabrication consulting transcripts and summaries. Built with hybrid search (dense + sparse), metadata filtering, and comprehensive evaluation framework.

## Features

### Core Capabilities
- **Hybrid Search**: Combines dense vector search (OpenAI embeddings) with sparse BM25 keyword matching
- **Type-Aware Chunking**: Intelligent chunking strategies for transcripts vs summaries
- **Rich Metadata Extraction**: Named entity recognition, keyword matching, action items, decisions
- **Query Processing**: Automatic query expansion, rewriting, and entity extraction
- **Metadata Filtering**: Filter by client, date range, document type, PowerFab modules
- **Evaluation Framework**: Built-in metrics (precision, recall, MRR, faithfulness, relevancy)
- **Interactive UI**: Streamlit-based chat interface with source citations

### Technology Stack
- **Vector Database**: Qdrant (local or cloud)
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions, can be reduced)
- **LLM**: OpenAI GPT-4 for response generation and query rewriting
- **Evaluation**: RAGAS framework for systematic quality assessment
- **NER**: spaCy for named entity extraction
- **UI**: Streamlit for interactive chat interface

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│  Documents → Chunking → Metadata → Embeddings → Qdrant         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                           │
│  Query → Processing → Hybrid Search → Fusion → Reranking       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   GENERATION PIPELINE                           │
│  Context Assembly → LLM Generation → Citations                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (for local Qdrant)
- OpenAI API key

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
│   │   ├── embedding_generator.py # OpenAI embeddings
│   │   └── document_loader.py     # File loading
│   ├── retrieval/                  # Search & query processing
│   │   ├── query_processor.py     # Query expansion/rewriting
│   │   └── hybrid_searcher.py     # Dense + sparse fusion
│   ├── generation/                 # Response generation
│   │   └── llm_interface.py       # LLM integration
│   ├── evaluation/                 # Quality metrics
│   │   └── metrics.py             # Precision, recall, RAGAS
│   ├── database/                   # Vector database
│   │   └── qdrant_client.py       # Qdrant operations
│   └── utils/                      # Utilities
│       ├── config_loader.py       # Configuration
│       └── text_utils.py          # Text processing
├── scripts/
│   └── ingest_documents.py        # CLI ingestion script
├── ui/
│   └── streamlit_app.py           # Streamlit interface
├── tests/                          # Unit & integration tests
├── .env                            # API keys (create from .env.example)
├── requirements.txt                # Python dependencies
├── Makefile                        # Common commands
└── README.md                       # This file
```

## Configuration

### Embedding Model Options

The system uses OpenAI embeddings by default. You can configure in `config/settings.yaml`:

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-large"  # or "text-embedding-3-small"
  dimensions: 3072  # 3072, 1536, 768, or 256
  batch_size: 100
```

**Cost Optimization:**
- `text-embedding-3-large` (3072 dim): $0.13/1M tokens - Best quality
- `text-embedding-3-large` (1536 dim): $0.13/1M tokens - Reduced dimensions
- `text-embedding-3-small` (1536 dim): $0.02/1M tokens - Lower cost

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

### Hybrid Search Fusion

```yaml
retrieval:
  fusion:
    method: "rrf"        # "rrf" or "weighted_sum"
    rrf_k: 60            # RRF constant
    dense_weight: 0.7    # For weighted_sum
    sparse_weight: 0.3   # For weighted_sum
```

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
- Embeddings: ~2M tokens × $0.13/1M = **$0.26**
- Total: **< $1**

**Per Query:**
- Query embedding: ~20 tokens × $0.13/1M = **$0.000003**
- LLM response: ~2000 tokens × $0.03/1K (GPT-4) = **$0.06**
- Total per query: **~$0.06**

**Monthly (100 queries):**
- ~$6/month

## Performance Benchmarks

Expected performance on M1 Mac:

- **Chunking**: >1000 chunks/second
- **Embedding**: ~100 docs/minute (API limited)
- **Ingestion**: ~50-100 docs/minute end-to-end
- **Query Latency**: <500ms (p95)
  - Dense search: <200ms
  - Sparse search: <100ms
  - Fusion: <50ms
  - LLM generation: 1-3 seconds

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

### v2.0 (2025-12-14)
- Complete rewrite with hybrid search
- OpenAI embeddings instead of Voyage-3/Cohere
- Type-aware chunking strategies
- Rich metadata extraction with NER
- Query processing pipeline
- Streamlit UI with filtering
- Evaluation framework with RAGAS
- Production-ready architecture

---

**Built with ❤️ for steel fabrication consulting**
