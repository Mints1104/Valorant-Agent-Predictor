# Valorant Composition & Win Prediction

A data science project. Analyzes VALORANT Champions Tour (VCT) esports data to study **agent composition synergy / meta trends** and build a **match win-prediction model**.

> Status: in progress. Data explored and cleaned, the main table is built, and line-ups
> have been turned into numbers ready for modelling. First model not fitted yet.

## Problem

Team comps in VALORANT esports shift with the patch meta, but it's not obvious which agent combinations actually correlate with winning at the pro level, or how much a draft (pick/ban) phase predicts a match outcome before a round is even played. This project looks at:

1. **Composition synergy / meta-analysis** — which agent combinations, per map, are over/under-performing relative to their pick rate.
2. **Win prediction** — given the draft (map picks/bans, agent comps), how well can a model predict the match winner?

## Data

Two Kaggle datasets, not committed to this repo (see [`data/README.md`](data/README.md) for how to fetch them via `kagglehub`):

| Dataset | Coverage | Role |
|---|---|---|
| VCT 2021-2026 Data | 2021–2026, all regions | Primary source — match results, agent picks, and the map pick/ban phase |
| VCT 2025 All Events (Int'l + Regional) | 2025 only | Supplies the match **dates**, which the primary source has none of. The two share the same match reference numbers |

Field-level definitions: [`columns_description.csv`](columns_description.csv).

## Scope

**2025 season only.** The earlier years have no date information anywhere, so their matches
can't be put in the order they were played — and that ordering is what stops the model
being tested unfairly (see "How the model is tested" below). Expanding to earlier seasons
is a possible later step once a working model exists; it would also mean handling agents
that didn't exist yet and maps that have since left the pool.

## Approach and progress

1. **Explore and clean** — done. See [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
2. **Build the main table** — done. One row per map played: [`notebooks/02_build_map_table.ipynb`](notebooks/02_build_map_table.ipynb), assembled by [`src/dataset.py`](src/dataset.py).
3. **Turn line-ups into numbers** — done. [`notebooks/03_agent_features.ipynb`](notebooks/03_agent_features.ipynb), built by [`src/features.py`](src/features.py).
4. **First model** — next. Logistic regression on agent picks, map, and who chose the map.
5. **Composition synergy** — which agent *combinations* do better than their individual parts suggest.
6. **Team strength** — each team's record going into a match, built only from their earlier matches.
7. **Write-up** — what worked, what didn't, and the limitations.

## What we've found so far

**Problems in the raw data** (all found by checking rather than assuming, and all fixed in `src/dataset.py`):

- The player file mixes whole-match summary rows in with the real per-map rows — 28% of it. Left in, they'd count many maps twice.
- That same file lists every player three times: once for the whole map, once for each half.
- NRG is stored as "Mega Minors" in the team columns but "NRG" everywhere else. Matching teams by name without fixing this silently loses all 25 of their matches.
- Five exhibition/all-star matches use made-up teams that never play again.
- The main dataset has no dates at all. They're taken from the second dataset, which shares the same match reference numbers — see [`data/README.md`](data/README.md).

**Findings so far:**

- Team A wins 51% of maps, so anything built has to beat a coin flip to be worth having.
- The team that chose the map wins it 53.8% of the time — real, but only about four extra wins per hundred maps. As a rule a model can be measured against, where it also has to answer on the 18% of maps nobody picked (fall back to team A), it gets 54.6% right.
- Which agents get played shifts heavily across the year. Tejo goes from one of the most-played agents to almost none; Omen rises from roughly one line-up in ten to one in six.
- **No single agent predicts winning.** Looking only at maps where one team had an agent and the other didn't, just 1 agent out of 27 has a win rate further from 50% than luck alone explains — and checking 27 things, that's exactly what chance produces. Teams also share about 3 of their 5 agents on a typical map, and on 99 maps the two line-ups were identical.

That last point sets expectations: a model built on agent picks alone should land near a coin flip. A high score would mean a mistake, not a discovery.

## How the model is tested

The season is split by **time**, not at random: learn from the 756 maps up to Masters
Toronto (late June), test on the 516 after it. The split falls in the gap between
tournaments, so no tournament is cut across both sides.

This matters because the game changes through the year. A random split would let the model
learn from September matches and be tested on March ones — it would already know how the
season turned out, so its score would look strong and mean nothing. This is the most likely
explanation for the 93% accuracy reported by one of the similar projects in
[`existing_projects.txt`](existing_projects.txt).

## Repo structure

```
Capstone-Project/
├── README.md                     # this file
├── project_idea_draft.txt        # original brainstorm notes
├── columns_description.csv       # data dictionary
├── requirements.txt              # project dependencies
├── test_data_pull.py             # smoke test: confirms kagglehub can fetch both datasets
├── data/
│   ├── README.md                 # how to fetch the raw data, and how dates are handled
│   └── VCT_2025_DATA_SUMMARY.md  # file-by-file notes on what each CSV contains
├── src/
│   ├── dates.py                  # match dates, bridged from the second dataset
│   ├── dataset.py                # builds the main table (one row per map played)
│   └── features.py               # turns line-ups into numbers a model can read
├── notebooks/
│   ├── 01_eda.ipynb              # exploring the raw files, and the problems in them
│   ├── 02_build_map_table.ipynb  # building and checking the main table
│   └── 03_agent_features.ipynb   # describing line-ups as numbers
├── Valorant_2025_All_Events_International_Regional/   # gitignored, local only
└── Valorant_Champion_Tour_2021-2026_Data/              # gitignored, local only
```

The `src/` modules are plain Python so the notebooks stay readable and the same logic can
be reused across them. Notebooks add `src/` to the import path themselves, so no install
step is needed.

Raw data folders exist locally but aren't tracked in git (see `.gitignore`) — several source CSVs exceed GitHub's 100MB file limit, so they're fetched on demand instead of versioned.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python test_data_pull.py      # confirms kagglehub can pull both datasets
```

See [`data/README.md`](data/README.md) for dataset-loading snippets.
