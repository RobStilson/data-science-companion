# Data Science Companion — Task List

## Legend
- [ ] pending
- [~] in_progress
- [x] completed

---

## Phase 1 — Foundation
- [x] T1: Project scaffold — requirements.txt, .env.example, agent/state.py, llm/factory.py, utils/formatting.py, test fixtures
  - Verify: ruff check . passes; pytest collects 0 errors (CHECKPOINT A)

## Phase 2 — Ingest
- [x] T2: F-01 File ingest node + Chainlit skeleton + tests/test_ingest.py
  - Verify: pytest tests/test_ingest.py -v

## Phase 3 — Pure Pandas Nodes
- [x] T3: F-03 Missing data node + F-05 Classification node + tests
  - Verify: pytest tests/test_missing.py tests/test_classify.py -v

## Phase 4 — Stats Nodes (CHECKPOINT B)
- [x] T4: F-04 Descriptive stats node + F-08 Skewness node + prompts/callouts.md + tests
  - Verify: pytest tests/test_descriptive.py tests/test_skewness.py -v

## Phase 5 — LLM-Heavy Nodes
- [ ] T5: F-02 Data dictionary node + prompts/data_dict.md + tests/test_data_dict.py
  - Verify: pytest tests/test_data_dict.py -v

## Phase 6 — User-Triggered Nodes
- [ ] T6: F-06 Correlation analysis node + tests/test_correlations.py
  - Verify: pytest tests/test_correlations.py -v
- [ ] T7: F-07 Visualization node + prompts/viz_suggest.md + tests/test_visualize.py
  - Verify: pytest tests/test_visualize.py -v

## Phase 7 — Export (CHECKPOINT C)
- [ ] T8: F-09 Code export — utils/code_gen.py + agent/nodes/export.py + tests/test_export.py
  - Verify: pytest tests/test_export.py -v

## Phase 8 — Integration
- [ ] T9: LangGraph graph wiring + full Chainlit app (agent/graph.py + app.py complete)
  - Verify: Manual smoke test — upload sample.csv, full EDA, histogram, export
- [ ] T10: Integration test + coverage >= 80% (tests/test_integration.py)
  - Verify: pytest --cov=agent --cov=utils; ruff check . (CHECKPOINT D)

---

## Post-Build Workflow
- [ ] /review — run code-review skill; address all findings
- [ ] /simplify — run simplify skill; remove unnecessary complexity
- [ ] /ship — lock requirements.txt, complete .env.example, write README
