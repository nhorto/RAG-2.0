# Quick Implementation Summary: Advanced Query Processing

**Reference:** See `PRD_ADVANCED_QUERY_PROCESSING.md` for complete specifications.

---

## Executive Decision: Multi-hop Reasoning

### ❌ DO NOT IMPLEMENT MULTI-HOP IN PHASE 1

**Rationale:**
- Only ~1-5% of consulting queries need multi-hop reasoning
- Adds 2-3x latency with compounding error rates
- Query augmentation + conversation history solves 95% of "multi-hop" use cases
- Re-evaluate after 3 months of production data

**Alternative:** Ship decomposition + augmentation first, measure real needs, iterate.

---

## What We're Building

### Three Core Features

1. **Query Decomposition** - Break "A, then B, then C" into separate queries
2. **Query Augmentation** - Add domain context to vague queries like "export this"
3. **Orchestration** - Coordinate multi-query retrieval and merge results

---

## Architecture Overview

```
User Query
    ↓
┌─────────────────────────────────────┐
│  1. QueryAnalyzer                   │  ← Heuristic detection (5ms)
│     - Needs decomposition?          │
│     - Needs augmentation?           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. QueryDecomposer (if needed)     │  ← LLM call (~800ms)
│     - Break into sub-queries        │
│     - Determine execution order     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. QueryAugmenter (if needed)      │  ← LLM call (~500ms)
│     - Add domain context            │
│     - Generate 2-5 variants         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Existing Processing             │
│     - Entity extraction             │
│     - Intent classification         │
│     - Abbreviation expansion        │
│     - Query rewriting               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. QueryOrchestrator               │
│     - Execute sub-queries           │
│     - Merge augmented variants      │
│     - Deduplicate results           │
└─────────────────────────────────────┘
    ↓
Final Results
```

---

## File Structure

### New Files (Create)
```
src/retrieval/
├── query_analyzer.py         # Detect if decomposition/augmentation needed
├── query_decomposer.py       # LLM-based query decomposition
├── query_augmenter.py        # Add domain context to vague queries
└── query_orchestrator.py     # Multi-query retrieval coordination

tests/
├── test_query_analyzer.py
├── test_query_decomposer.py
├── test_query_augmenter.py
└── test_query_orchestrator.py

data/test_queries/
└── advanced_test_queries.json
```

### Modified Files
```
src/retrieval/query_processor.py  # Add analyzer, decomposer, augmenter
config/settings.yaml               # Add feature flags
requirements.txt                   # Add tenacity
```

### Unchanged Files
```
src/retrieval/hybrid_searcher.py  # No changes needed
src/retrieval/reranker.py          # No changes needed
src/database/qdrant_client.py     # No changes needed
```

---

## Implementation Phases

### Phase 1: Query Decomposition (Week 1-2)
- **Build:** `QueryAnalyzer` + `QueryDecomposer`
- **Integrate:** Modify `QueryProcessor.process()`
- **Test:** 20 complex multi-part queries
- **Target:** 90% detection accuracy, <1.5s P95 latency

### Phase 2: Query Augmentation (Week 3)
- **Build:** `QueryAugmenter` + enhance `QueryAnalyzer`
- **Test:** 30 vague queries
- **Target:** 80% detection accuracy, +15% precision improvement

### Phase 3: Orchestration (Week 4)
- **Build:** `QueryOrchestrator` for multi-query retrieval
- **Integrate:** End-to-end pipeline
- **Test:** Full test suite + performance benchmarks
- **Target:** All latency targets met, no regressions

### Phase 4: Multi-hop (Optional, Week 5-6)
- **Decision Point:** Only if >10% of production queries fail without it
- **Wait:** 3 months of production data before deciding

---

## Key Detection Logic

### When to Decompose?

**Confidence >= 0.6 if:**
- Sequential words: "and then", "after that", "followed by" → +0.4
- Multiple "and" conjunctions → +0.3
- Multiple question marks → +0.4
- Enumeration: "first, second, third" → +0.3
- High clause count (≥3) → +0.2

**Example:** "How to create BOM, assign to WO, and track production?"
→ Decomposed into 3 sequential sub-queries

### When to Augment?

**Confidence >= 0.5 if:**
- Vague pronouns: "this", "that", "it" → +0.4
- Generic terms: "issue", "feature", "module" → +0.2
- Very short query (≤3 words) → +0.3
- Incomplete action: "export", "create" without object → +0.3

**Example:** "How to export this?"
→ Augmented to ["export BOM", "export Work Order", "export reports"]

---

## Example Query Flows

### Example 1: Complex Multi-Part Query

**Input:** "How do I create a BOM, assign it to a WO, and track production?"

**Processing:**
1. Analyzer: `needs_decomposition=True` (confidence=0.7)
2. Decomposer: Breaks into 3 sub-queries:
   - "How to create a Bill of Materials in PowerFab Estimating?"
   - "How to assign a BOM to a Work Order in PowerFab?"
   - "How to track production status in PowerFab Production Control?"
3. Orchestrator: Execute sequentially, merge results
4. **Total time:** ~2.5s | **LLM calls:** 1 (GPT-4)

### Example 2: Vague Query

**Input:** "How to export this?"

**Processing:**
1. Analyzer: `needs_augmentation=True` (confidence=0.7)
2. Augmenter: Generate variants:
   - "How to export BOM reports from PowerFab Estimating?"
   - "How to export Work Orders from PowerFab Production Control?"
   - "How to export material lists from PowerFab Purchasing?"
3. Orchestrator: Search with all variants, merge results
4. **Total time:** ~1.2s | **LLM calls:** 1 (GPT-3.5)

### Example 3: Simple Query (No Enhancement)

**Input:** "How to create a Bill of Materials in Estimating module?"

**Processing:**
1. Analyzer: `needs_decomposition=False`, `needs_augmentation=False`
2. Standard pipeline: Entity extraction → Intent classification → Retrieval
3. **Total time:** ~0.5s | **LLM calls:** 0 (existing rewriting only)

---

## Performance Targets

| Query Type | Mean Latency | P95 Latency | LLM Calls | Cost/Query |
|-----------|--------------|-------------|-----------|------------|
| Simple | <100ms | <200ms | 0 | $0.000 |
| Augmented | <700ms | <1000ms | 1 (GPT-3.5) | $0.0004 |
| Decomposed | <1000ms | <1500ms | 1 (GPT-4) | $0.012 |
| Complex | <1500ms | <2500ms | 2 | $0.0124 |

**Monthly Cost Impact (1000 queries):** +$2.20/month (~3% increase from $60 → $62)

---

## Configuration

### Feature Flags (settings.yaml)

```yaml
query_processing:
  advanced_processing:
    # Enable/disable features
    enable_decomposition: true
    enable_augmentation: true

    # Thresholds
    analysis:
      min_decompose_confidence: 0.6
      min_augment_confidence: 0.5

    # Model selection
    decomposition:
      llm_model: "gpt-4"           # Complex reasoning
      temperature: 0.3
      max_sub_queries: 5

    augmentation:
      llm_model: "gpt-3.5-turbo"   # Cheaper model
      temperature: 0.5
      max_variants: 5

    # Performance
    orchestration:
      parallel_workers: 5
      enable_deduplication: true
```

---

## Testing Strategy

### Unit Tests (Per Component)
- `test_query_analyzer.py` - Detection heuristics
- `test_query_decomposer.py` - LLM decomposition + fallback
- `test_query_augmenter.py` - Domain augmentation
- `test_query_orchestrator.py` - Merge logic + deduplication

### Integration Tests
- End-to-end: Complex query → decomposition → retrieval → results
- End-to-end: Vague query → augmentation → retrieval → results
- Regression: Simple queries still work (no overhead)

### Performance Benchmarks
- Latency measurements for all query types
- Cost tracking per query
- Memory usage profiling

### Test Dataset
Create `data/test_queries/advanced_test_queries.json` with:
- 20 complex multi-part queries (for decomposition)
- 30 vague queries (for augmentation)
- 50 simple queries (regression tests)

---

## Success Criteria

### Phase 1 (Decomposition)
- ✅ 90% of multi-part queries correctly detected
- ✅ Decomposition latency <1.5s (P95)
- ✅ No regression on simple queries

### Phase 2 (Augmentation)
- ✅ 80% of vague queries correctly detected
- ✅ Precision@10 improvement +15% on vague queries
- ✅ Cost per augmented query <$0.001

### Phase 3 (Orchestration)
- ✅ All query types handled correctly
- ✅ Deduplication working (no duplicate chunks)
- ✅ All latency targets met
- ✅ No breaking changes to existing API

---

## Rollout Plan

### Week 1: Internal Testing
- Deploy with feature flags OFF
- Enable for development environment only
- Validate all components working

### Week 2: Canary (10% traffic)
- Enable for 10% of production queries
- Monitor latency, error rates, costs
- Collect user feedback

### Week 3: Gradual Rollout (50%)
- Increase to 50% if metrics look good
- Continue monitoring

### Week 4: Full Rollout (100%)
- Enable for all users if no issues
- Document final metrics

### Rollback Triggers
- Error rate >10%
- P95 latency >3s
- Cost explosion (>$0.02/query)
- User complaints about quality

---

## Error Handling

### Fallback Strategy
| Error | Fallback |
|-------|----------|
| LLM API timeout | Use original query without enhancement |
| Invalid JSON response | Retry once, then fallback |
| Too many sub-queries | Truncate to 5 |
| All sub-queries fail | Fall back to original query |
| Zero augmentation variants | Use original query |

### Retry Logic
- Use `tenacity` library with exponential backoff
- Max 2 retry attempts per LLM call
- Timeout: 10 seconds per call

---

## Open Questions (For Implementation)

1. **Caching strategy:** Should we cache LLM decomposition/augmentation results?
   - **Recommendation:** Yes, use LRU cache with 1-hour TTL (20-30% hit rate expected)

2. **Conversation history:** Should we add session context for pronoun resolution?
   - **Recommendation:** Phase 2 enhancement, not Phase 1 (wait for user need)

3. **Batch processing:** Can we batch multiple user queries for efficiency?
   - **Recommendation:** Not needed for single-user consulting system (low QPS)

4. **Model fine-tuning:** Should we fine-tune GPT-3.5 for domain-specific augmentation?
   - **Recommendation:** Evaluate after 3 months if cost becomes issue

---

## Next Steps (After PRD Approval)

1. **Create implementation tickets** for each phase
2. **Set up test environment** with feature flags
3. **Create test dataset** (`advanced_test_queries.json`)
4. **Begin Phase 1 development** (QueryAnalyzer + QueryDecomposer)
5. **Schedule weekly check-ins** to review progress

---

**For detailed specifications, see:** `PRD_ADVANCED_QUERY_PROCESSING.md`

**Questions?** Review the PRD or consult with the architect.
