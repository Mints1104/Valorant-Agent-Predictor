# Data sources

Raw data is **not** committed to this repo (see `.gitignore`) — it's pulled from Kaggle at run time with `kagglehub`. This keeps the repo small and avoids GitHub's 100MB per-file limit, which several of the raw CSVs here exceed.

## Setup

```bash
pip install kagglehub[pandas-datasets]
```

You'll need a Kaggle account and API token (`~/.kaggle/kaggle.json`) the first time you run this — `kagglehub` will prompt for it.

## Dataset 1 — VCT 2025, all events (international + regional)

Kaggle dataset: [`piyush86kumar/valorant-vct-2025-all-events`](https://www.kaggle.com/datasets/piyush86kumar/valorant-vct-2025-all-events) — "Valorant 2025 - All Events International + Regional"

Local folder this maps to: `Valorant_2025_All_Events_International_Regional/`

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "piyush86kumar/valorant-vct-2025-all-events",
    "VCT 2025 Americas Stage 1_csvs/matches.csv",  # path within the dataset
)
```

## Dataset 2 — VCT 2021-2026

Kaggle dataset: [`ryanluong1/valorant-champion-tour-2021-2023-data`](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) — the dataset's slug still says `2021-2023` (Kaggle doesn't rename slugs when a dataset is updated), but the page title is "Valorant Champion Tour 2021-2026 Data" and it now covers 2021-2026. Use the slug below as-is.

Local folder this maps to: `Valorant_Champion_Tour_2021-2026_Data/`

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "vct_2025/agents/teams_picked_agents.csv"  # path within the dataset

df = kagglehub.dataset_load(
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
