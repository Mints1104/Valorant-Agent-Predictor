# Data sources

Raw data is **not** committed to this repo (see `.gitignore`) — it's pulled from Kaggle at run time with `kagglehub`. This keeps the repo small and avoids GitHub's 100MB per-file limit, which several of the raw CSVs here exceed.

## Setup

```bash
pip install kagglehub[pandas-datasets]
```

You'll need a Kaggle account and an API token the first time you run this.

Kaggle now issues a single API token rather than the older username-plus-key pair. Generate
one at [kaggle.com/settings/api](https://www.kaggle.com/settings/api) ("API Tokens →
Generate New Token"), then save the token on its own in a plain text file at
`~/.kaggle/access_token` (on Windows, `C:\Users\<you>\.kaggle\access_token`). `kagglehub`
picks it up from there automatically — no environment variables needed.

Two things that don't work, and cost time if you try them:

- Putting the token in a `.env` file. Nothing here reads `.env`, so it's ignored silently.
- Naming the variable `KAGGLE_API_KEY`. If you'd rather use an environment variable than a
  file, the name `kagglehub` actually looks for is `KAGGLE_API_TOKEN`.

The older `~/.kaggle/kaggle.json` still works if you already have one.

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

## Match dates (the two datasets share a match ID space)

Dataset 2 has **no date column in any of its 21 tables** — verified by scanning
every file. That's a problem, because a chronological train/test split is what
stops patch/meta leakage between train and test.

Dataset 1 supplies the missing dates. Both are scraped from vlr.gg, so the IDs
line up: `match_id` in Dataset 1 is the same value as `Match ID` in Dataset 2's
`ids/tournaments_stages_matches_games_ids.csv`. Verified for 2025 — all 503
Dataset 2 matches get a date, every tournament at 100% coverage.

Use [`src/dates.py`](../src/dates.py):

```python
from dates import load_match_dates

dates = load_match_dates()   # match_id, match_date, match_datetime, event, stage, week
```

Two gotchas it handles, both worth knowing about:

- **8 dates don't parse raw.** vlr.gg shows a "Today"/"Yesterday" badge on recent
  matches and the scrape glued it onto the string (`"Sun, October 5, 2025Today"`).
  Stripping that suffix leaves 0 failures, and weekday names then agree with the
  parsed dates on all 504 rows.
- **Ties.** 10 rows share an exact timestamp (concurrent matches on different
  streams), so sort by `(match_datetime, match_id)` for a stable order.

`match_id` correlates with time (Spearman 0.979) but is **not** monotonic — it
decreases against the clock in 182 places — so order by the date, not the ID.

### 2024 dates

`load_match_dates("vct_2024")` reads dates for the 2024 season from a folder called
`vct_2024/` at the repo root. It is gitignored like all raw data. It comes from the same
author as the 2025 date source and has the same layout: one folder per event, each with a
`matches.csv`. It dates all 434 of the main dataset's 2024 matches, every event at 100%.

Unlike the rest of the data, it is not downloaded through `kagglehub`. Put the folder in
place by hand. `load_match_dates` raises a clear error, naming the folder, if it is missing.

The same ID trap applies: in 2024, `match_id` runs against the clock in 160 places.

## Column reference

See [`columns_description.csv`](../columns_description.csv) (tracked in the repo) for the field-by-field dictionary — it's small and mostly still applies to Dataset 2's equivalent tables, though some column names differ slightly (check headers when in doubt).
