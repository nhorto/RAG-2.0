# Query Processing Decision Tree

**Quick reference for understanding query routing and feature activation.**

---

## Decision Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INCOMING QUERY                               │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  1. QUERY ANALYSIS   │
                    │   (QueryAnalyzer)    │
                    │  • Check complexity  │
                    │  • Check specificity │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐
      │  Decomposition?     │     │   Augmentation?     │
      │  (confidence ≥0.6)  │     │  (confidence ≥0.5)  │
      └──────────┬──────────┘     └──────────┬──────────┘
                 │                           │
    ┌────────────┴────────────┐  ┌──────────┴───────────┐
    ▼                         ▼  ▼                      ▼
  [YES]                     [NO] [YES]                [NO]
    │                         │  │                      │
    ▼                         │  ▼                      │
┌─────────────────────┐      │ ┌──────────────────┐   │
│ 2. DECOMPOSE QUERY  │      │ │ 3. AUGMENT QUERY │   │
│  (QueryDecomposer)  │      │ │ (QueryAugmenter) │   │
│  • LLM call (GPT-4) │      │ │ • LLM call (3.5) │   │
│  • ~800ms latency   │      │ │ • ~500ms latency │   │
└──────────┬──────────┘      │ └────────┬─────────┘   │
           │                 │          │             │
           └─────────────┐   │   ┌──────┘             │
                         ▼   ▼   ▼                    ▼
                    ┌────────────────────────────────────┐
                    │   4. EXISTING PROCESSING           │
                    │   • Entity extraction              │
                    │   • Intent classification          │
                    │   • Abbreviation expansion         │
                    │   • Query rewriting (existing LLM) │
                    └─────────────┬──────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
    ┌──────────────────────┐          ┌───────────────────┐
    │  5a. ORCHESTRATOR    │          │  5b. SIMPLE PATH  │
    │  (if decomposed OR   │          │  (if neither)     │
    │   augmented)         │          │                   │
    │  • Multi-query       │          │  • Single search  │
    │  • Merge results     │          │  • Standard flow  │
    │  • Deduplicate       │          │                   │
    └──────────┬───────────┘          └─────────┬─────────┘
               │                                 │
               └─────────────┬───────────────────┘
                             ▼
                   ┌──────────────────┐
                   │  HYBRID SEARCHER │
                   │  (existing)      │
                   └────────┬─────────┘
                            ▼
                   ┌──────────────────┐
                   │  FINAL RESULTS   │
                   └──────────────────┘
```

---

## Detection Logic Details

### Decomposition Detection (Confidence Scoring)

```
Score = 0.0

IF "and then" OR "after that" OR "followed by" in query:
    Score += 0.4 ✅ STRONG SIGNAL

IF " and " OR ", and " in query:
    Score += 0.3 ⚠️ MODERATE SIGNAL

IF multiple "?" in query:
    Score += 0.4 ✅ STRONG SIGNAL

IF "first" OR "second" OR "1." OR "2." in query:
    Score += 0.3 ⚠️ MODERATE SIGNAL

IF clause_count >= 3:  # Count of commas, "and", "or", "then"
    Score += 0.2 ⚠️ WEAK SIGNAL

──────────────────────────────────────
IF Score >= 0.6:
    DECOMPOSE = TRUE ✅
ELSE:
    DECOMPOSE = FALSE ❌
```

### Augmentation Detection (Confidence Scoring)

```
Score = 0.0

IF "this" OR "that" OR "it" in query:
    Score += 0.4 ✅ STRONG SIGNAL

IF "issue" OR "problem" OR "thing" OR "feature" in query:
    Score += 0.2 ⚠️ MODERATE SIGNAL

IF word_count <= 3:
    Score += 0.3 ⚠️ MODERATE SIGNAL

IF ("export" OR "import" OR "create" OR "delete") AND word_count <= 4:
    Score += 0.3 ⚠️ MODERATE SIGNAL (incomplete action)

──────────────────────────────────────
IF Score >= 0.5:
    AUGMENT = TRUE ✅
ELSE:
    AUGMENT = FALSE ❌
```

---

## Query Categorization Examples

### Category 1: Simple Query (No Enhancement)

**Query:** "How to create a Bill of Materials in the Estimating module?"

**Analysis:**
- Decomposition score: 0.0 (no multi-part indicators)
- Augmentation score: 0.0 (specific, complete)

**Route:** ❌ No decomposition, ❌ No augmentation
**Path:** Direct → Existing processing → Hybrid search
**Latency:** ~500ms
**LLM calls:** 0 (existing rewriting only)

---

### Category 2: Augmented Query

**Query:** "How to export this?"

**Analysis:**
- Decomposition score: 0.0
- Augmentation score: 0.7 (vague pronoun "this" +0.4, short query +0.3)

**Route:** ❌ No decomposition, ✅ Augmentation
**Path:** Augment → 3-5 variants → Orchestrator → Merge
**Variants Generated:**
1. "How to export BOM reports from PowerFab Estimating?"
2. "How to export Work Orders from PowerFab Production Control?"
3. "How to export material lists from PowerFab Purchasing?"

**Latency:** ~1.2s
**LLM calls:** 1 (GPT-3.5-turbo for augmentation)

---

### Category 3: Decomposed Query

**Query:** "How do I create a BOM, assign it to a WO, and track production?"

**Analysis:**
- Decomposition score: 0.7 (conjunction " and " +0.3, 3 clauses +0.2, sequential flow +0.2)
- Augmentation score: 0.0

**Route:** ✅ Decomposition, ❌ No augmentation
**Path:** Decompose → 3 sub-queries → Sequential retrieval → Merge
**Sub-Queries Generated:**
1. "How to create a Bill of Materials in PowerFab Estimating?"
2. "How to assign a BOM to a Work Order in PowerFab?"
3. "How to track production status in PowerFab Production Control?"

**Execution:** Sequential (each query runs in order)
**Latency:** ~2.5s (decomposition 800ms + 3x retrieval 600ms each)
**LLM calls:** 1 (GPT-4 for decomposition)

---

### Category 4: Complex Query (Both Enhancements)

**Query:** "How to fix this issue with BOM creation, WO assignment, and tracking?"

**Analysis:**
- Decomposition score: 0.8 (3 topics + conjunction)
- Augmentation score: 0.6 (vague pronoun "this" + generic term "issue")

**Route:** ✅ Decomposition, ✅ Augmentation
**Path:** Decompose AND Augment → Orchestrator → Merge all variants
**Processing:**
1. Decompose into 3 sub-queries
2. Augment each sub-query with domain context
3. Execute all variants (parallel where possible)
4. Merge and deduplicate results

**Latency:** ~3.0s (parallel decompose+augment 800ms + retrieval 1.5s + merge 200ms)
**LLM calls:** 2 (1 GPT-4 for decomposition + 1 GPT-3.5 for augmentation)

---

## Orchestration Strategy by Query Type

### Sequential Decomposition (AND/SEQUENTIAL logic)

```
Sub-query 1 → Search → Results A
                ↓
Sub-query 2 → Search → Results B
                ↓
Sub-query 3 → Search → Results C
                ↓
         Interleave: [A1, B1, C1, A2, B2, C2, ...]
                ↓
         Deduplicate → Final Results
```

**Use when:** Steps are related (BOM → WO → Production)

---

### Parallel Decomposition (OR logic)

```
Sub-query 1 ──┐
              ├→ Search in parallel
Sub-query 2 ──┤
              │
Sub-query 3 ──┘
     ↓
Union all results → Sort by score → Deduplicate
     ↓
Final Results
```

**Use when:** Alternatives (issues with BOM OR issues with WO)

---

### Augmented Query Execution

```
Variant 1 ──┐
            │
Variant 2 ──┤→ Search in parallel (top_k=5 each)
            │
Variant 3 ──┘
     ↓
Union all results → Sort by score → Deduplicate
     ↓
Final Results (top_k=10)
```

**Strategy:** Cast wide net with variants, then rank and deduplicate

---

## Performance Characteristics

| Query Type | Detection Time | LLM Time | Retrieval Time | Total P95 |
|-----------|----------------|----------|----------------|-----------|
| Simple | 5ms | 0ms | 500ms | <800ms |
| Augmented | 5ms | 500ms | 700ms | <1.5s |
| Decomposed | 5ms | 800ms | 900ms | <2.0s |
| Complex | 5ms | 1200ms | 1200ms | <3.0s |

**Key Insight:** Only complex queries (2% of traffic) pay the full latency cost.

---

## Cost Breakdown

| Component | LLM Model | Tokens (avg) | Cost/Call |
|-----------|-----------|--------------|-----------|
| Decomposition | GPT-4 | 400 | $0.012 |
| Augmentation | GPT-3.5-turbo | 200 | $0.0004 |
| Analysis | None (heuristic) | 0 | $0.00 |
| Orchestration | None | 0 | $0.00 |

**Average cost per query:** ~$0.0022 (weighted by query type distribution)

---

## Feature Flag Configuration

```yaml
# Quick config reference
query_processing:
  advanced_processing:
    enable_decomposition: true   # Turn on/off decomposition
    enable_augmentation: true    # Turn on/off augmentation

    analysis:
      min_decompose_confidence: 0.6  # Threshold (0.0-1.0)
      min_augment_confidence: 0.5    # Threshold (0.0-1.0)
```

**Pro tip:** Start with higher thresholds (0.7, 0.6) to be conservative, then lower based on production data.

---

## Debugging Guide

### Query not being decomposed when it should?

1. Check detection score: Log `analysis.decomposition_confidence`
2. If score < threshold, check for keywords:
   - Missing "and then" or sequential indicators?
   - Try adding explicit conjunctions
3. If score high but still not decomposing:
   - Check feature flag `enable_decomposition`
   - Check LLM API key is valid

### Query not being augmented when it should?

1. Check detection score: Log `analysis.augmentation_confidence`
2. If score < threshold:
   - Query might be specific enough (false negative)
   - Consider lowering threshold for your domain
3. If augmented but poor variants:
   - Check domain vocabulary is loaded
   - Review augmentation system prompt
   - Consider fine-tuning GPT-3.5

### Results worse than before?

1. Check for over-decomposition:
   - Lower `min_decompose_confidence` threshold
   - Review decomposition prompts
2. Check for over-augmentation:
   - Augmentation may add noise for specific queries
   - Consider skipping augmentation for queries >10 words
3. Check deduplication:
   - Ensure orchestrator is deduplicating by `chunk_id`

---

## Quick Test Commands

```python
# Test detection logic
from src.retrieval.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()
analysis = analyzer.analyze("How to export this?")
print(f"Decompose: {analysis.needs_decomposition} ({analysis.decomposition_confidence:.2f})")
print(f"Augment: {analysis.needs_augmentation} ({analysis.augmentation_confidence:.2f})")
```

```python
# Test full pipeline
from src.retrieval.query_processor import QueryProcessor

processor = QueryProcessor()
result = processor.process("How to create BOM, assign to WO, and track?")

if result.decomposed:
    print(f"Decomposed into {len(result.decomposed.sub_queries)} sub-queries:")
    for sq in result.decomposed.sub_queries:
        print(f"  {sq.order}: {sq.query_text}")

if result.augmented:
    print(f"Augmented into {len(result.augmented.augmented_variants)} variants:")
    for v in result.augmented.augmented_variants:
        print(f"  - {v}")
```

---

**For complete implementation details, see:** `PRD_ADVANCED_QUERY_PROCESSING.md`
