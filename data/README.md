# Data sources

Raw data is **not** committed to this repo (see `.gitignore`) — it's pulled from Kaggle at run time with `kagglehub`. This keeps the repo small and avoids GitHub's 100MB per-file limit, which several of the raw CSVs here exceed.

## Setup

```bash
pip install kagglehub[pandas-datasets]
```

You'll need a Kaggle account and API token (`~/.kaggle/kaggle.json`) the first time you run this — `kagglehub` will prompt for it.

## Dataset 1 — VCT 2025, all events (international + regional)

Kaggle dataset: `<TODO: paste the exact slug from the dataset's Kaggle page — check the "Copy API command" button>`

Local folder this maps to: `Valorant_2025_All_Events_International_Regional/`

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "<owner>/<dataset-slug>",
    "<event folder>/<file>.csv",  # e.g. "VCT 2025 Americas Stage 1_csvs/matches.csv"
)
```

## Dataset 2 — VCT 2021-2026

Kaggle dataset: `ryanluong1/valorant-champion-tour-2021-2023-data`

> **Verify before using:** this slug is the 2021-2023 version. The local folder here is named `2021-2026`, which suggests a newer/renamed version of the same dataset — check the dataset's Kaggle page and update the slug above if it differs.

Local folder this maps to: `Valorant_Champion_Tour_2021-2026_Data/`

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "vct_2025/agents/teams_picked_agents.csv"  # path within the dataset

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ryanluong1/valorant-champion-tour-2021-2023-data",
    file_path,
)

print(df.head())
```

Key files for the composition-synergy / win-prediction angle:
- `vct_<year>/matches/draft_phase.csv` — pick/ban phase per map per team
- `vct_<year>/agents/teams_picked_agents.csv` — team + map + agent picks with wins/losses attached
- `vct_<year>/matches/overview.csv` — per-player, per-map stats (rating, ACS, KAST, etc.)
- `vct_<year>/matches/maps_scores.csv`, `scores.csv` — match/map results for the win-prediction target

Avoid loading `kills.csv` / `rounds_kills.csv` for 2021-2022 unless you need round-by-round detail — they're the largest files (120-150MB each) and are overkill for match/map-level analysis.

## Column reference

See [`columns_description.csv`](../columns_description.csv) (tracked in the repo) for the field-by-field dictionary — it's small and mostly still applies to Dataset 2's equivalent tables, though some column names differ slightly (check headers when in doubt).
