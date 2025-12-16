# Advanced Query Processing System

**Version:** 1.0
**Date:** 2025-12-15
**Status:** ✅ Implementation Complete

---

## Overview

The Advanced Query Processing System enhances TeklaPowerFabRAG_v2 with intelligent query decomposition and augmentation capabilities. This system automatically detects when queries need special handling and routes them through appropriate processing pipelines.

### Key Features

✅ **Query Decomposition** - Breaks multi-part queries into atomic sub-queries
✅ **Query Augmentation** - Adds domain context to vague/underspecified queries
✅ **Intelligent Routing** - Heuristic-based detection (fast, no LLM overhead)
✅ **Graceful Fallbacks** - Robust error handling with fallback to simple queries
✅ **Feature Flags** - Gradual rollout with configuration control

---

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│  QueryAnalyzer (~5ms)               │  ← Heuristic detection
│  - Decomposition confidence         │
│  - Augmentation confidence          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  QueryDecomposer (conditional)      │  ← GPT-4 (~800ms)
│  - Break into sub-queries           │
│  - Determine execution order        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  QueryAugmenter (conditional)       │  ← GPT-3.5 (~500ms)
│  - Add domain context               │
│  - Generate 2-5 variants            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Traditional Processing             │
│  - Entity extraction                │
│  - Intent classification            │
│  - Query expansion                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  QueryOrchestrator (if needed)      │
│  - Execute sub-queries              │
│  - Merge results                    │
│  - Deduplicate chunks               │
└─────────────────────────────────────┘
```

---

## Components

### 1. QueryAnalyzer

**File:** `src/retrieval/query_analyzer.py`
**Purpose:** Fast heuristic detection of query complexity

**Detection Heuristics:**

| Feature | Signal | Confidence Boost |
|---------|--------|------------------|
| Sequential words ("and then", "after") | Decomposition | +0.4 |
| Multiple "and" conjunctions | Decomposition | +0.3 |
| Multiple question marks | Decomposition | +0.4 |
| Enumeration ("1.", "first") | Decomposition | +0.3 |
| High clause count (≥3) | Decomposition | +0.2 |
| Vague pronouns ("this", "that") | Augmentation | +0.4 |
| Generic terms ("issue", "feature") | Augmentation | +0.2 |
| Short queries (≤3 words) | Augmentation | +0.3 |
| Incomplete actions ("export") | Augmentation | +0.3 |

**Thresholds:**
- Decomposition: confidence ≥ 0.6
- Augmentation: confidence ≥ 0.5

### 2. QueryDecomposer

**File:** `src/retrieval/query_decomposer.py`
**Purpose:** LLM-based decomposition of complex queries

**Features:**
- Uses GPT-4 for complex reasoning
- Generates 2-5 atomic sub-queries
- Determines connection logic (AND/OR/SEQUENTIAL)
- Preserves domain context
- Exponential backoff retry with tenacity
- Fallback to original query on failure

**Example:**
```
Input:  "How to create BOM, assign to WO, and track production?"
Output:
  1. "How to create a Bill of Materials in PowerFab Estimating?"
  2. "How to assign a BOM to a Work Order in PowerFab?"
  3. "How to track production status in PowerFab Production Control?"

Logic: SEQUENTIAL
Strategy: sequential
```

### 3. QueryAugmenter

**File:** `src/retrieval/query_augmenter.py`
**Purpose:** Add domain context to vague queries

**Features:**
- Uses GPT-3.5-turbo (cost-effective)
- Generates 2-5 specific variants
- Resolves pronouns to domain entities
- Completes incomplete actions
- Rule-based fallback if LLM fails

**Example:**
```
Input:  "How to export this?"
Output:
  1. "How to export BOM reports from PowerFab Estimating?"
  2. "How to export Work Orders from PowerFab Production Control?"
  3. "How to export material lists from PowerFab Purchasing?"
```

### 4. QueryOrchestrator

**File:** `src/retrieval/query_orchestrator.py`
**Purpose:** Coordinate multi-query retrieval and merge results

**Features:**
- Parallel execution (ThreadPoolExecutor)
- Sequential execution (for workflow queries)
- Result interleaving (AND/SEQUENTIAL logic)
- Result ranking (OR logic)
- Chunk deduplication by ID

**Execution Strategies:**
- **Parallel:** Independent sub-queries (OR logic)
- **Sequential:** Step-by-step workflows (SEQUENTIAL logic)
- **Interleaved:** Multiple related topics (AND logic)

---

## Configuration

### Settings Location
`config/settings.yaml` → `query_processing.advanced_processing`

### Key Configuration Options

```yaml
advanced_processing:
  # Master switch
  enabled: true

  # Feature flags
  enable_decomposition: true
  enable_augmentation: true

  # Thresholds
  analysis:
    min_decompose_confidence: 0.6
    min_augment_confidence: 0.5

  # Models
  decomposition:
    llm_model: "gpt-4"
    temperature: 0.3
    max_sub_queries: 5

  augmentation:
    llm_model: "gpt-3.5-turbo"
    temperature: 0.5
    max_variants: 5

  # Orchestration
  orchestration:
    parallel_workers: 5
    enable_deduplication: true
```

### Feature Flag Control

**Disable all advanced processing:**
```yaml
advanced_processing:
  enabled: false
```

**Disable only decomposition:**
```yaml
advanced_processing:
  enabled: true
  enable_decomposition: false
  enable_augmentation: true
```

**Adjust thresholds for more conservative detection:**
```yaml
analysis:
  min_decompose_confidence: 0.7  # Higher = fewer decompositions
  min_augment_confidence: 0.6    # Higher = fewer augmentations
```

---

## Usage

### Basic Usage

```python
from retrieval.query_processor import QueryProcessor

# Initialize processor
processor = QueryProcessor()

# Process query
result = processor.process("How to create BOM, assign to WO, and track?")

# Check what happened
if result.decomposed:
    print(f"Decomposed into {len(result.decomposed.sub_queries)} sub-queries")
    for sq in result.decomposed.sub_queries:
        print(f"  {sq.order}. {sq.query_text}")

if result.augmented:
    print(f"Augmented into {len(result.augmented.augmented_variants)} variants")
    for variant in result.augmented.augmented_variants:
        print(f"  - {variant}")

# Get all query variants for retrieval
all_variants = result.all_query_variants
print(f"Total variants: {len(all_variants)}")
```

### With Orchestrator

```python
from retrieval.query_processor import QueryProcessor
from retrieval.query_orchestrator import QueryOrchestrator
from retrieval.hybrid_searcher import HybridSearcher

# Initialize components
processor = QueryProcessor()
searcher = HybridSearcher()
orchestrator = QueryOrchestrator(searcher)

# Process query
processed = processor.process("How to create BOM and assign to WO?")

# Execute retrieval
if processed.decomposed:
    results = orchestrator.execute_decomposed(
        processed.decomposed,
        filters=processed.filter,
        final_top_k=20
    )
    print(f"Retrieved {len(results.merged_results)} results")
    print(f"Execution time: {results.execution_time_ms:.1f}ms")

elif processed.augmented:
    results = orchestrator.execute_augmented(
        processed.augmented,
        filters=processed.filter,
        final_top_k=10
    )
    print(f"Retrieved {len(results)} results from augmented variants")
```

---

## Testing

### Run Unit Tests

```bash
# Test analyzer only
pytest tests/test_query_analyzer.py -v

# Test integration (requires OPENAI_API_KEY)
pytest tests/test_advanced_query_integration.py -v

# Run all tests
pytest tests/ -v
```

### Test Dataset

Location: `data/test_queries/advanced_test_queries.json`

**Contents:**
- 20 test queries covering all scenarios
- Expected decomposition/augmentation flags
- Ground truth topics
- Complexity classifications

**Categories:**
- Simple queries (no enhancement)
- Decomposition queries (multi-part)
- Augmentation queries (vague)
- Complex queries (both enhancements)
- Edge cases

### Manual Testing

```python
# Test analyzer
from retrieval.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()
analysis = analyzer.analyze("How to export this?")

print(f"Needs decomposition: {analysis.needs_decomposition}")
print(f"Needs augmentation: {analysis.needs_augmentation}")
print(f"Complexity: {analysis.query_complexity}")
print(f"Reasons: {analysis.augmentation_reasons}")
```

---

## Performance

### Latency Targets

| Query Type | Mean Latency | P95 Latency | LLM Calls |
|-----------|--------------|-------------|-----------|
| Simple | <100ms | <200ms | 0 |
| Augmented | <700ms | <1000ms | 1 (GPT-3.5) |
| Decomposed | <1000ms | <1500ms | 1 (GPT-4) |
| Complex | <1500ms | <2500ms | 2 |

### Cost Impact

**Per Query:**
- Simple: $0.000
- Augmented: ~$0.0004
- Decomposed: ~$0.012
- Complex: ~$0.0124

**Monthly (1000 queries):**
- Baseline: ~$60/month
- With advanced processing: ~$62/month (+3%)

### Query Distribution (Expected)

- Simple: 75% (no overhead)
- Augmented: 20% (low cost)
- Decomposed: 4% (higher cost)
- Complex: 1% (highest cost)

**Weighted average cost:** ~$0.0022/query

---

## Troubleshooting

### Query Not Being Decomposed

**Issue:** Multi-part query not decomposing when expected

**Solutions:**
1. Check confidence score in logs
2. Lower threshold in config:
   ```yaml
   analysis:
     min_decompose_confidence: 0.5  # Was 0.6
   ```
3. Verify sequential indicators are present
4. Check feature flag is enabled

### Query Not Being Augmented

**Issue:** Vague query not being augmented

**Solutions:**
1. Check confidence score
2. Lower threshold:
   ```yaml
   analysis:
     min_augment_confidence: 0.4  # Was 0.5
   ```
3. Verify vague pronouns or short length
4. Check feature flag is enabled

### LLM Failures

**Issue:** Decomposition/augmentation failing

**Solutions:**
1. Check OpenAI API key is set
2. Verify API quota/rate limits
3. Check network connectivity
4. System falls back to original query automatically

### Poor Quality Results

**Issue:** Results worse than before advanced processing

**Solutions:**
1. Disable specific feature:
   ```yaml
   enable_decomposition: false  # Or enable_augmentation: false
   ```
2. Increase confidence thresholds (be more conservative)
3. Review decomposed sub-queries in logs
4. Check if deduplication is removing good results

---

## Rollout Plan

### Phase 1: Internal Testing (Week 1)
- ✅ Deploy with `enabled: false`
- ✅ Enable in development environment
- ✅ Validate all components working
- ✅ Run comprehensive test suite

### Phase 2: Canary Rollout (Week 2)
- Enable for 10% of production queries
- Monitor latency, error rates, costs
- Collect user feedback
- Adjust thresholds based on data

### Phase 3: Gradual Rollout (Week 3)
- Increase to 50% if metrics good
- Continue monitoring
- Fine-tune configuration

### Phase 4: Full Rollout (Week 4)
- Enable for 100% of users
- Document final metrics
- Create runbook for operations

### Rollback Triggers
- Error rate >10%
- P95 latency >3s
- Cost explosion (>$0.02/query)
- User complaints about quality

---

## Future Enhancements

### Phase 4: Multi-hop Reasoning (Optional)
**Decision:** Wait 3 months for production data

**Criteria to implement:**
- >10% of queries fail without it
- Clear user need for follow-up questions
- Cost justification established

### Caching Layer
**Implementation:** LRU cache with 1-hour TTL

**Expected Benefits:**
- 20-30% cache hit rate
- Reduce LLM costs
- Improve latency

### Session Context
**Feature:** Use conversation history for pronoun resolution

**Example:**
```
User: "How to create a BOM?"
...
User: "How to export it?"  ← Resolve "it" to "BOM" from history
```

---

## Maintenance

### Monitoring

**Key Metrics:**
- Detection accuracy (decomposition/augmentation)
- LLM call success rate
- Latency percentiles (P50, P95, P99)
- Cost per query
- User satisfaction ratings

**Dashboards:**
- Query complexity distribution
- Enhancement trigger rates
- Error rates and failure modes
- Cost trends over time

### Log Analysis

```python
# Example log entry structure
{
  "query": "How to create BOM and assign WO?",
  "analysis": {
    "needs_decomposition": true,
    "decomposition_confidence": 0.7,
    "complexity": "complex"
  },
  "decomposed": {
    "sub_queries": 2,
    "logic": "SEQUENTIAL"
  },
  "latency_ms": 1234,
  "cost_usd": 0.012
}
```

### Performance Tuning

**If latency too high:**
1. Lower confidence thresholds (fewer enhancements)
2. Reduce max_sub_queries and max_variants
3. Increase parallel_workers for orchestration
4. Enable caching (future)

**If cost too high:**
1. Raise confidence thresholds (more selective)
2. Consider fine-tuning GPT-3.5 for augmentation
3. Implement caching to reduce duplicate calls

---

## References

- **PRD:** `docs/PRD_ADVANCED_QUERY_PROCESSING.md`
- **Implementation Summary:** `docs/IMPLEMENTATION_SUMMARY_ADVANCED_QUERY.md`
- **Decision Tree:** `docs/QUERY_PROCESSING_DECISION_TREE.md`
- **Test Dataset:** `data/test_queries/advanced_test_queries.json`

---

## Support

For questions or issues:
1. Check this README and PRD documentation
2. Review test examples in `tests/`
3. Check configuration in `config/settings.yaml`
4. Consult decision tree for query routing logic

---

**Last Updated:** 2025-12-15
**Implementation Status:** ✅ Complete
**Production Ready:** ✅ Yes (pending rollout)
