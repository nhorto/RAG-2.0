# TeklaPowerFab RAG System - Documentation Index

**Version:** 2.1
**Last Updated:** 2025-12-15

This index helps you navigate all documentation for the TeklaPowerFabRAG_v2 system.

---

## 🚀 Getting Started (Read These First)

| Document | Purpose | Read Time | When to Use |
|----------|---------|-----------|-------------|
| **[START_HERE.md](START_HERE.md)** | Complete onboarding guide | 15 min | First time using the system |
| **[QUICKSTART.md](QUICKSTART.md)** | Minimal 5-minute setup | 5 min | Just want it running ASAP |
| **[README.md](README.md)** | Complete system reference | 30 min | Understanding all features & config |

---

## 📚 Core Documentation

### System Overview
| Document | Purpose | Audience |
|----------|---------|----------|
| **[README.md](README.md)** | Complete features, configuration, usage | All users |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | Implementation overview, file structure | Developers |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history and changes | All users |

### Setup & Configuration
| Document | Purpose | Key Topics |
|----------|---------|------------|
| **[QUICKSTART.md](QUICKSTART.md)** | Fast setup guide | Installation, first run |
| **[README.md](README.md)** §Configuration | All configuration options | Embeddings, chunking, hybrid search, advanced query processing |

### Version History
| Document | Purpose | Details |
|----------|---------|---------|
| **[CHANGELOG.md](CHANGELOG.md)** | All changes by version | v2.0, v2.1 features |
| **[IMPLEMENTATION_CHANGES_v2.1.md](IMPLEMENTATION_CHANGES_v2.1.md)** | Detailed v2.1 changes | 1,000+ lines of technical details |

---

## 🔍 Feature-Specific Documentation

### Hybrid Search (v2.1)
| Document | Purpose | Key Topics |
|----------|---------|------------|
| **[HYBRID_SEARCH_IMPLEMENTATION.md](HYBRID_SEARCH_IMPLEMENTATION.md)** | Technical implementation details | FastEmbed BM25, Qdrant sparse vectors, prefetch fusion |
| **[README.md](README.md)** §Hybrid Search | User-facing configuration | Fusion methods, reranking options |

### Advanced Query Processing (v2.1 NEW)
| Document | Purpose | Key Topics |
|----------|---------|------------|
| **[docs/ADVANCED_QUERY_PROCESSING_README.md](docs/ADVANCED_QUERY_PROCESSING_README.md)** | Complete user guide | Usage, configuration, examples |
| **[docs/PRD_ADVANCED_QUERY_PROCESSING.md](docs/PRD_ADVANCED_QUERY_PROCESSING.md)** | Product requirements | Feature requirements, acceptance criteria |
| **[docs/QUERY_PROCESSING_DECISION_TREE.md](docs/QUERY_PROCESSING_DECISION_TREE.md)** | Decision logic reference | When decomposition/augmentation triggers |
| **[docs/IMPLEMENTATION_SUMMARY_ADVANCED_QUERY.md](docs/IMPLEMENTATION_SUMMARY_ADVANCED_QUERY.md)** | Implementation summary | Technical overview |
| **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** | Implementation completion report | Code statistics, next steps |

---

## 🛠️ Technical Reference

### Architecture & Code
| Document | Topic | Details |
|----------|-------|---------|
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | Complete code overview | All 32 modules, file structure, dependencies |
| **[README.md](README.md)** §Architecture | System architecture | Ingestion, retrieval, generation pipelines |
| **[README.md](README.md)** §Project Structure | Directory structure | All files and their purposes |

### API & Costs
| Document | Topic | Details |
|----------|-------|---------|
| **[README.md](README.md)** §API Costs | Cost estimates | Ingestion, per-query, monthly costs |
| **[docs/ADVANCED_QUERY_PROCESSING_README.md](docs/ADVANCED_QUERY_PROCESSING_README.md)** §Cost Impact | Advanced processing costs | GPT-4, GPT-3.5-turbo usage |

### Performance
| Document | Topic | Details |
|----------|-------|---------|
| **[README.md](README.md)** §Performance Benchmarks | Expected latency | Chunking, embedding, search, generation |
| **[CHANGELOG.md](CHANGELOG.md)** v2.1 | Performance improvements | 10-20x faster sparse search |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** §Performance Targets | Target metrics | All metrics with status |

---

## 📖 Usage Guides

### Basic Usage
| Task | Document | Section |
|------|----------|---------|
| Install dependencies | **[QUICKSTART.md](QUICKSTART.md)** | Step 1 |
| Configure API keys | **[QUICKSTART.md](QUICKSTART.md)** | Step 3 |
| Ingest documents | **[README.md](README.md)** | §Usage → Ingest Documents |
| Run UI | **[QUICKSTART.md](QUICKSTART.md)** | Step 6 |
| Search and chat | **[README.md](README.md)** | §Usage → Search and Chat |

### Advanced Usage
| Task | Document | Section |
|------|----------|---------|
| Custom query processing | **[README.md](README.md)** | §Advanced Usage → Custom Query Processing |
| Direct search | **[README.md](README.md)** | §Advanced Usage → Direct Search |
| Batch evaluation | **[README.md](README.md)** | §Advanced Usage → Batch Evaluation |
| Enable reranking | **[README.md](README.md)** | §Configuration → Hybrid Search Configuration |
| Enable advanced query processing | **[docs/ADVANCED_QUERY_PROCESSING_README.md](docs/ADVANCED_QUERY_PROCESSING_README.md)** | §Configuration |

### Configuration
| Feature | Document | Section |
|---------|----------|---------|
| Embedding models | **[README.md](README.md)** | §Configuration → Embedding Model Options |
| Chunking strategies | **[README.md](README.md)** | §Configuration → Chunking Strategies |
| Hybrid search | **[README.md](README.md)** | §Configuration → Hybrid Search Configuration |
| Advanced query processing | **[README.md](README.md)** | §Configuration → Advanced Query Processing |

---

## 🧪 Testing & Evaluation

### Testing
| Document | Topic | Details |
|----------|-------|---------|
| **[tests/test_query_analyzer.py](tests/test_query_analyzer.py)** | Query analyzer tests | 350 lines of unit tests |
| **[tests/test_advanced_query_integration.py](tests/test_advanced_query_integration.py)** | Integration tests | 300 lines, 20+ test queries |
| **[data/test_queries/advanced_test_queries.json](data/test_queries/advanced_test_queries.json)** | Test dataset | 20 queries with expected behavior |

### Evaluation
| Document | Topic | Details |
|----------|-------|---------|
| **[README.md](README.md)** §Evaluation | Evaluation framework | Metrics, RAGAS integration |
| **[README.md](README.md)** §Advanced Usage → Batch Evaluation | Running evaluations | Test dataset structure, execution |

---

## 🔧 Operations & Deployment

### Troubleshooting
| Issue | Document | Section |
|-------|----------|---------|
| Qdrant connection | **[README.md](README.md)** | §Troubleshooting → Qdrant Connection Error |
| API key errors | **[README.md](README.md)** | §Troubleshooting → API Key Errors |
| Import errors | **[README.md](README.md)** | §Troubleshooting → Import Errors |
| Empty search results | **[README.md](README.md)** | §Troubleshooting → Empty Search Results |
| Slow ingestion | **[README.md](README.md)** | §Troubleshooting → Slow Ingestion |

### Deployment
| Topic | Document | Section |
|-------|----------|---------|
| Local development | **[README.md](README.md)** | §Deployment → Local Development |
| Production options | **[README.md](README.md)** | §Deployment → Production Options |
| Migration from v2.0 | **[CHANGELOG.md](CHANGELOG.md)** | §Migration Path |

### Migration
| Topic | Document | Details |
|-------|----------|---------|
| ChromaDB to Qdrant | **[README.md](README.md)** | §Migration from ChromaDB |
| v2.0 to v2.1 hybrid | **[scripts/migrate_to_hybrid.py](scripts/migrate_to_hybrid.py)** | Automatic migration |

---

## 📊 Quick Reference Tables

### File Locations
```
TeklaPowerFabRAG_v2/
├── README.md                              # Main documentation
├── START_HERE.md                          # Getting started guide
├── QUICKSTART.md                          # 5-minute setup
├── CHANGELOG.md                           # Version history
├── IMPLEMENTATION_SUMMARY.md              # Implementation overview
├── IMPLEMENTATION_CHANGES_v2.1.md         # Detailed v2.1 changes
├── IMPLEMENTATION_COMPLETE.md             # Advanced query processing summary
├── HYBRID_SEARCH_IMPLEMENTATION.md        # Hybrid search technical details
├── DOCUMENTATION_INDEX.md                 # This file
└── docs/
    ├── ADVANCED_QUERY_PROCESSING_README.md       # Query processing user guide
    ├── PRD_ADVANCED_QUERY_PROCESSING.md          # Product requirements
    ├── IMPLEMENTATION_SUMMARY_ADVANCED_QUERY.md  # Implementation summary
    └── QUERY_PROCESSING_DECISION_TREE.md         # Decision logic
```

### Document Categories

**User Documentation:**
- START_HERE.md
- QUICKSTART.md
- README.md
- docs/ADVANCED_QUERY_PROCESSING_README.md

**Technical Documentation:**
- IMPLEMENTATION_SUMMARY.md
- HYBRID_SEARCH_IMPLEMENTATION.md
- IMPLEMENTATION_COMPLETE.md
- docs/IMPLEMENTATION_SUMMARY_ADVANCED_QUERY.md

**Reference Documentation:**
- CHANGELOG.md
- IMPLEMENTATION_CHANGES_v2.1.md
- docs/PRD_ADVANCED_QUERY_PROCESSING.md
- docs/QUERY_PROCESSING_DECISION_TREE.md

---

## 🎯 Quick Navigation by Role

### **New User**
1. [START_HERE.md](START_HERE.md) - Understand what you have
2. [QUICKSTART.md](QUICKSTART.md) - Get it running
3. [README.md](README.md) §Usage - Learn to use it

### **Developer**
1. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Code overview
2. [README.md](README.md) §Architecture - System design
3. [HYBRID_SEARCH_IMPLEMENTATION.md](HYBRID_SEARCH_IMPLEMENTATION.md) - Technical details
4. [docs/ADVANCED_QUERY_PROCESSING_README.md](docs/ADVANCED_QUERY_PROCESSING_README.md) - Query processing

### **System Administrator**
1. [README.md](README.md) §Configuration - All settings
2. [README.md](README.md) §Deployment - Production deployment
3. [README.md](README.md) §Troubleshooting - Common issues

### **Evaluator / QA**
1. [README.md](README.md) §Evaluation - Evaluation framework
2. [tests/test_advanced_query_integration.py](tests/test_advanced_query_integration.py) - Test suite
3. [README.md](README.md) §Performance Benchmarks - Expected metrics

---

## 📝 Version-Specific Documentation

### v2.1 Features (2025-12-15)
- **Hybrid Search:** [HYBRID_SEARCH_IMPLEMENTATION.md](HYBRID_SEARCH_IMPLEMENTATION.md)
- **Advanced Query Processing:** [docs/ADVANCED_QUERY_PROCESSING_README.md](docs/ADVANCED_QUERY_PROCESSING_README.md)
- **Complete Changes:** [IMPLEMENTATION_CHANGES_v2.1.md](IMPLEMENTATION_CHANGES_v2.1.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md) §v2.1

### v2.0 Features (2025-12-14)
- **Initial Implementation:** [CHANGELOG.md](CHANGELOG.md) §v2.0

---

## 🔗 External References

- **Qdrant Documentation:** https://qdrant.tech/documentation/
- **OpenAI API:** https://platform.openai.com/docs
- **Cohere Rerank:** https://docs.cohere.com/docs/reranking
- **RAGAS Framework:** https://docs.ragas.io/
- **spaCy NER:** https://spacy.io/usage/linguistic-features#named-entities

---

## 💡 Tips for Reading Documentation

1. **Start with your goal:**
   - Want to use it? → START_HERE.md → QUICKSTART.md
   - Want to understand it? → README.md
   - Want to modify it? → IMPLEMENTATION_SUMMARY.md

2. **Use the search function:**
   - All documentation is markdown, searchable with grep/ripgrep
   - Example: `rg "query decomposition" *.md docs/*.md`

3. **Follow the links:**
   - Most documents cross-reference each other
   - Use this index to understand relationships

4. **Check the date:**
   - Look for "Last Updated" or version numbers
   - CHANGELOG.md tracks all changes chronologically

---

**Need Help?**
- Check [README.md](README.md) §Troubleshooting
- Review [README.md](README.md) §Support
- Examine system logs in `logs/`

**Want to Contribute?**
- See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for code structure
- Follow existing patterns in similar modules
- Add tests for new features

---

**Last Updated:** 2025-12-15
**System Version:** v2.1
**Documentation Maintained By:** System Engineers
