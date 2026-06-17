# Data Science Companion — Build Plan

## Dependency Graph

```
[T1 Foundation] AgentState + llm/factory + utils/formatting + fixtures
    └── [T2] ingest.py (F-01)  — all nodes need a DataFrame
            ├── [T3] missing.py (F-03) + classify.py (F-05)  — pure pandas, no LLM
            │       └── [T6] correlations.py (F-06)  — needs classify col lists
            ├── [T4] descriptive.py (F-04) + skewness.py (F-08)  — scipy + LLM callouts
            ├── [T5] data_dict.py (F-02)  — LLM-heavy, needs prompts/
            ├── [T7] visualize.py (F-07)  — Plotly + LLM suggestions
            └── [T8] export.py (F-09) + utils/code_gen.py  — depends on session_log
                    └── [T9] graph.py + app.py  — wires everything in Chainlit
                            └── [T10] integration test + coverage check
```

---

## T1 — Project Scaffold
**Status:** pending
**Files:** requirements.txt, .env.example, agent/state.py, llm/factory.py,
utils/formatting.py, tests/fixtures/sample.csv, tests/fixtures/sample.xlsx

- agent/state.py — AgentState TypedDict: df, filename, file_size_mb, session_log,
  cat_cols, num_cols, discrete_cols, continuous_cols, outcome_col, messages
- llm/factory.py — reads LLM_PROVIDER env var; returns ChatAnthropic | ChatOpenAI |
  ChatOllama; raises ValueError for unknown provider
- utils/formatting.py — make_markdown_table(), severity_label(), skew_label()
- fixtures — synthetic ~200-row CSV/XLSX: mixed types, nulls, skewed col, binary col

Verify: ruff check . passes; pytest tests/ -x collects with 0 errors.
CHECKPOINT A

---

## T2 — F-01: File Ingest + Chainlit Skeleton
**Status:** pending
**Files:** agent/nodes/ingest.py, app.py (skeleton), tests/test_ingest.py

- Auto-detect CSV vs Excel by extension
- Multi-sheet Excel: store sheet names, ask user to pick before loading
- >100 MB: warn + ask to confirm before loading
- Unsupported extension: clear error, no state update
- Success: store df + filename + file_size_mb; send shape + head(5) markdown preview

Verify: pytest tests/test_ingest.py -v

---

## T3 — F-03: Missing Data + F-05: Classification
**Status:** pending
**Files:** agent/nodes/missing.py, agent/nodes/classify.py,
tests/test_missing.py, tests/test_classify.py

- missing.py — null count/pct + severity_label per col; overall completeness %;
  zero-null cols excluded from table but noted in summary; appends "missing_data" to session_log
- classify.py — categorical: object/category OR numeric <=20 unique; discrete: <10 unique;
  continuous: >=10 unique; stores col lists in state; appends "classify" to session_log
- Edge cases: all-null col, boundaries at 10 and 20 unique values, numeric-only dataset

Verify: pytest tests/test_missing.py tests/test_classify.py -v

---

## T4 — F-04: Descriptive Statistics + F-08: Skewness
**Status:** pending
**Files:** agent/nodes/descriptive.py, agent/nodes/skewness.py,
prompts/callouts.md, tests/test_descriptive.py, tests/test_skewness.py

- descriptive.py — df.describe() on numeric cols; IQR outlier detection; std>mean flag;
  min==max flag; passes findings JSON to LLM (prompts/callouts.md) for plain-language summary;
  always emits callouts section; appends "descriptive" to session_log
- prompts/callouts.md — prompt receiving callout JSON dict, returns concise paragraph
- skewness.py — scipy.stats.skew per numeric col; sorted by |skew| desc; skew_label;
  transformation suggestions for High-skew cols (log/sqrt/Box-Cox); no LLM; appends "skewness"

Verify: pytest tests/test_descriptive.py tests/test_skewness.py -v
CHECKPOINT B

---

## T5 — F-02: Data Dictionary
**Status:** pending
**Files:** agent/nodes/data_dict.py, prompts/data_dict.md, tests/test_data_dict.py

- Single LLM call batching all columns; prompt returns JSON {"col": "description", ...}
- Table: Column | Type | Samples (3) | Nulls | Null% | Description
- Appends "data_dict" to session_log
- Unit test mocks LLM; validates all columns present; descriptions are single sentences

Verify: pytest tests/test_data_dict.py -v

---

## T6 — F-06: Correlation Analysis
**Status:** pending
**Files:** agent/nodes/correlations.py, tests/test_correlations.py

- Reads state.outcome_col (set by message handler in T9)
- Method auto-selection per predictor (shown to user):
  - outcome binary (nunique==2) -> Point-Biserial
  - predictor |skew|>1 or ordinal -> Spearman
  - else -> Pearson
- Table: Predictor | Correlation | Method | p-value | Strength; sorted by |corr| desc
- Top 5 positive + top 5 negative bold-highlighted
- Unknown outcome col -> helpful error, state unchanged
- Appends "correlations" to session_log

Verify: pytest tests/test_correlations.py -v

---

## T7 — F-07: Data Visualization
**Status:** pending
**Files:** agent/nodes/visualize.py, prompts/viz_suggest.md, tests/test_visualize.py

- LLM generates chart suggestions after classify; cached in state
- Supported types: histogram, box, scatter, bar, heatmap, line, pair plot
- Renders interactive chart via cl.Plotly; unknown col or unsupported type -> clear error
- Each viz appends {"viz": type, "cols": [...]} to session_log

Verify: pytest tests/test_visualize.py -v

---

## T8 — F-09: Code Export
**Status:** pending
**Files:** utils/code_gen.py, agent/nodes/export.py, tests/test_export.py

- code_gen.py:
  - generate_python(session_log, filename) -> str — self-contained .py; DATA_PATH placeholder
  - generate_r(session_log, filename) -> str — equivalent R; library(tidyverse) + ggplot2
- export.py — writes temp files; attaches via cl.File; filenames analysis_export.py + .R
- Appends "export" to session_log

Verify: pytest tests/test_export.py -v
CHECKPOINT C

---

## T9 — LangGraph Graph + Full Chainlit App
**Status:** pending
**Files:** agent/graph.py, app.py (complete)

- graph.py — StateGraph(AgentState); auto-run: ingest -> data_dict -> missing ->
  classify -> descriptive -> skewness; conditional: correlations if outcome_col set;
  viz and export on-demand by user messages
- app.py:
  - on_chat_start — welcome + upload instructions
  - on_file_upload — trigger ingest; run linear EDA through skewness; send correlation prompt
  - on_message — route: outcome col answer | viz request | export request

Verify: Manual smoke test — upload sample.csv; walk full EDA; request histogram;
request export; confirm both files non-empty.

---

## T10 — Integration Test + Coverage
**Status:** pending
**Files:** tests/test_integration.py

- Full LangGraph run against sample.csv with real LLM (not mocked)
- Assert session_log contains all 5 auto-run steps
- Set outcome_col to binary col; run correlations; assert ranked table in messages
- pytest --cov=agent --cov=utils >= 80% coverage
- ruff check . zero warnings

CHECKPOINT D — proceed to /review, /simplify, /ship
