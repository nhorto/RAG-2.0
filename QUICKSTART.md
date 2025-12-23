# Tekla PowerFab RAG System - Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
cd /Users/nicholashorton/Documents/TeklaPowerFabRAG_v2

# Install Python packages and spaCy model
make install
```

## Step 2: Start Qdrant (1 minute)

```bash
# Start Qdrant in Docker
make setup
```

This creates:
- Qdrant container running on port 6333
- `.env` file for your API keys

## Step 3: Configure API Key (30 seconds)

Edit `.env` and add your OpenAI API key:

```bash
# Open in your editor
nano .env

# Add your key:
OPENAI_API_KEY=sk-your-actual-key-here
```

Save and exit (Ctrl+X, then Y, then Enter).

## Step 4: Prepare Your Documents (1 minute)

**Option A: Symlink (Recommended)**
```bash
ln -s "/Users/nicholashorton/Documents/LLM Sumarization" data/raw
```

**Option B: Copy Files**
```bash
mkdir -p data/raw
cp "/Users/nicholashorton/Documents/LLM Sumarization"/*.txt data/raw/
```

## Step 5: Ingest Documents (1-5 minutes depending on number of docs)

```bash
make ingest
```

You'll see:
- Documents being loaded
- Chunks being created
- Embeddings being generated
- Upload to Qdrant

## Step 6: Launch the UI (10 seconds)

```bash
make run-ui
```

Your browser will open to http://localhost:8501

## Step 7: Try It Out!

**Example Questions:**

1. "How do I create a BOM in the Estimating module?"
2. "What did we discuss about Production Control last week?"
3. "Show me all issues with Work Orders"
4. "What decisions were made about shipping workflows?"

**Use Filters:**
- Enter a client name
- Select a date range
- Choose document type
- Pick search mode (hybrid recommended)

## Troubleshooting

### "Connection refused" or "Cannot connect to Qdrant"

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker start qdrant

# Or restart from scratch
make setup
```

### "OpenAI API key not found"

```bash
# Verify your .env file has the key
cat .env

# Should show:
# OPENAI_API_KEY=sk-...

# If not, edit it:
nano .env
```

### "No results found"

Collection might be empty:

```bash
# Re-ingest documents
make ingest

# Or check collection status in Qdrant
docker logs qdrant
```

### Import errors or missing packages

```bash
# Reinstall everything
make install

# Or manually install specific packages
pip install openai qdrant-client streamlit
python -m spacy download en_core_web_sm
```

## Next Steps

### Add More Documents

1. Copy new `.txt` or `.srt` files to `data/raw/`
2. Run `make ingest` again
3. Refresh the UI

### Optimize for Cost

Edit `config/settings.yaml`:

```yaml
embeddings:
  model: "text-embedding-3-small"  # Cheaper model
  dimensions: 1536  # Smaller dimensions
```

Then re-ingest.

### Customize Chunking

Edit `config/settings.yaml`:

```yaml
chunking:
  transcript:
    chunk_size: 768  # Larger chunks
    overlap: 75      # More overlap
```

### View All Settings

```bash
cat config/settings.yaml
```

## Daily Usage

### Start Everything
```bash
# If Qdrant stopped
docker start qdrant

# Launch UI
make run-ui
```

### Stop Everything
```bash
# Stop UI: Ctrl+C in terminal

# Stop Qdrant (optional)
make stop-qdrant
```

### Update Documents
```bash
# Add new files to data/raw/
# Then:
make ingest
```

## Getting Help

1. Check the full README.md for detailed documentation
2. Review troubleshooting section above
3. Check Qdrant logs: `docker logs qdrant`
4. Check system logs: `tail -f logs/rag_system.log`

## Congratulations! 🎉

You now have a production-ready RAG system for your consulting transcripts!

**What you can do:**
- ✅ Search across all your transcripts and summaries
- ✅ Filter by client, date, document type
- ✅ Get AI-generated answers with citations
- ✅ View source documents for verification
- ✅ Track conversation history

**Enjoy your enhanced knowledge retrieval!**