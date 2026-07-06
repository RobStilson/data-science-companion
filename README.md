# Data Science Companion

A conversational AI agent for exploratory data analysis. Upload a CSV or Excel file and get a complete EDA — data dictionary, missing-data report, descriptive stats, skewness, correlation analysis, interactive charts, and downloadable Python/R scripts — all inside a Chainlit chat session.

## Features

| Step | What it produces |
|---|---|
| File upload | Shape confirmation, column preview, snake_case column normalisation |
| Data dictionary | dtype, sample values, null count/%, LLM-generated description per column |
| Missing data | Per-column null counts with Low / Medium / High severity labels |
| Descriptive stats | `df.describe()` + high-dispersion, zero-variance, and IQR outlier callouts |
| Classification | Columns split into categorical, discrete numeric, and continuous numeric |
| Skewness report | Per-column skew labels + suggested transformations for highly skewed columns |
| Auto visualisation | Histograms, bar charts, and a correlation heatmap generated automatically on upload |
| On-demand charts | `histogram`, `box`, `scatter`, `bar`, `heatmap`, `line`, `pair plot` via chat command |
| Correlation analysis | Pearson / Spearman / Point-Biserial ranked table for a named outcome column |
| Code export | Runnable `.py` (pandas + Plotly) and `.R` (tidyverse + ggplot2) scripts, downloadable inline |

## Tech stack

- **Agent** — LangGraph (each analysis step is a graph node)
- **LLM** — provider-agnostic via `llm/factory.py`; supports Anthropic, OpenAI, Ollama
- **UI** — Chainlit (file upload, chat, inline Plotly charts, file downloads)
- **Data** — pandas, openpyxl
- **Visualisation** — Plotly (interactive), Matplotlib/Seaborn (export scripts)
- **Statistics** — scipy, statsmodels
- **Tests** — pytest + pytest-asyncio (unit + integration)
- **Linting** — ruff

## Prerequisites

- Python 3.11+
- An API key for your chosen LLM provider (Anthropic, OpenAI, or a running Ollama instance)

## Setup

```bash
git clone https://github.com/RobStilson/data-science-companion.git
cd data-science-companion
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the env template and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_PROVIDER=anthropic          # anthropic | openai | ollama
LLM_MODEL=claude-sonnet-4-6     # model name for the chosen provider
ANTHROPIC_API_KEY=sk-...        # only the key that matches LLM_PROVIDER
```

## Running

```bash
python start.py
```

Then open [http://localhost:8080](http://localhost:8080).

`start.py` applies a compatibility patch for Python 3.14 + `nest_asyncio` + `anyio` before launching Chainlit. You can also pass `--port`:

```bash
python start.py --port 8888
```

## Usage

1. Upload a CSV or Excel file using the attachment button.
2. The agent runs the full auto-EDA pipeline and posts results section by section.
3. Use the action buttons or type commands in chat:

| Command | Example |
|---|---|
| Chart — single column | `histogram age` |
| Chart — two columns | `scatter age vs salary` or `scatter age and salary` |
| Correlation heatmap | `heatmap` |
| Correlation analysis | `correlate on salary` |
| Suggest charts | `suggest charts` |
| Export scripts | `export` |

## Running tests

```bash
pytest
```

Target coverage: >= 80% on `agent/` and `utils/`. The integration test (`tests/test_integration.py`) runs the full LangGraph pipeline against `tests/fixtures/sample.csv` without mocking.

## Project structure

```
data-science-companion/
├── app.py                  # Chainlit entry point
├── start.py                # Launcher with Python 3.14 / asyncio compatibility patch
├── .env.example
├── requirements.txt
├── SPEC.md
│
├── agent/
│   ├── graph.py            # LangGraph state graph
│   ├── state.py            # AgentState TypedDict
│   └── nodes/
│       ├── ingest.py       # File loading + snake_case column normalisation
│       ├── missing.py      # Missing data report
│       ├── classify.py     # Categorical / discrete / continuous classification
│       ├── descriptive.py  # Descriptive statistics + callouts
│       ├── skewness.py     # Skewness report
│       ├── data_dict.py    # Data dictionary (LLM descriptions)
│       ├── auto_viz.py     # Automatic chart generation on upload
│       ├── visualize.py    # On-demand chart rendering
│       ├── correlations.py # Correlation analysis
│       └── export.py       # Python + R script generation
│
├── llm/
│   └── factory.py          # Returns a configured ChatModel from env vars
│
├── prompts/
│   ├── data_dict.md
│   ├── callouts.md
│   └── viz_suggest.md
│
├── utils/
│   ├── formatting.py       # Markdown table builders, severity labels
│   └── code_gen.py         # Python + R script template generators
│
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── sample.csv
    │   └── sample.xlsx
    └── test_*.py           # One file per agent node + integration
```
