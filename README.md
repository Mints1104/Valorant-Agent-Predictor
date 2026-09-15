# Valorant Composition & Win Prediction

A data science project. Analyzes VALORANT Champions Tour (VCT) esports data to study **agent composition synergy / meta trends** and build a **match win-prediction model**.

> Status: kickoff — project starts 2026-09-16. This README will be updated as the analysis develops.

## Problem

Team comps in VALORANT esports shift with the patch meta, but it's not obvious which agent combinations actually correlate with winning at the pro level, or how much a draft (pick/ban) phase predicts a match outcome before a round is even played. This project looks at:

1. **Composition synergy / meta-analysis** — which agent combinations, per map, are over/under-performing relative to their pick rate.
2. **Win prediction** — given the draft (map picks/bans, agent comps), how well can a model predict the match winner?

## Data

Two Kaggle datasets, not committed to this repo (see [`data/README.md`](data/README.md) for how to fetch them via `kagglehub`):

| Dataset | Coverage | Role |
|---|---|---|
| VCT 2021-2026 Data | 2021–2026, all regions | Primary source — has dedicated `draft_phase.csv` (pick/ban) and `teams_picked_agents.csv` (comp + win/loss) tables |
| VCT 2025 All Events (Int'l + Regional) | 2025 only | Smaller, cleaner subset for quick prototyping/sanity checks |

Field-level definitions: [`columns_description.csv`](columns_description.csv).

## Planned approach

1. **Explore & clean** — pull the relevant tables via `kagglehub`, check for missing/inconsistent team & agent naming across years (agent balance changes, team rebrands).
2. **Feature engineering** — per-match team composition vectors, historical team/agent win rates, map-specific comp performance, draft order.
3. **Composition synergy analysis** — which agent pairs/comps outperform their expected win rate given individual agent pick/win rates (association-rule or lift-style analysis).
4. **Win-prediction model** — baseline (logistic regression) vs a stronger model (gradient boosting), evaluated on held-out later tournaments to avoid meta leakage across patches.
5. **Write-up** — findings on synergy patterns and model performance/limitations.

## Repo structure

```
Capstone-Project/
├── README.md                  # this file
├── project_idea_draft.txt     # original brainstorm notes
├── columns_description.csv    # data dictionary
├── requirements.txt           # project dependencies
├── test_data_pull.py          # smoke test: confirms kagglehub can fetch both datasets
├── data/
│   └── README.md              # how to fetch the raw datasets via kagglehub
├── notebooks/                 # exploration / analysis notebooks
├── Valorant_2025_All_Events_International_Regional/   # gitignored, local only
└── Valorant_Champion_Tour_2021-2026_Data/              # gitignored, local only
```

Raw data folders exist locally but aren't tracked in git (see `.gitignore`) — several source CSVs exceed GitHub's 100MB file limit, so they're fetched on demand instead of versioned.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python test_data_pull.py      # confirms kagglehub can pull both datasets
```

See [`data/README.md`](data/README.md) for dataset-loading snippets.
