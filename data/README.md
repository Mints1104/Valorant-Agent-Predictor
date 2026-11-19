# The data

The raw files download through `kagglehub` and are cached on your
machine. The versions are pinned, so after the first download everything works offline.

## Downloading

All three datasets are public, so no Kaggle account or token is needed. Run
`python test_data_pull.py` to download them (about 31 MB) and check they load.

## The datasets

| Dataset | Version | Gives us |
|---|---|---|
| [`ryanluong1/valorant-champion-tour-2021-2023-data`](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) | 47 | Results, line-ups, map picks and player stats for 2021–2026 (the name was never updated). No dates |
| [`piyush86kumar/valorant-vct-2025-all-events`](https://www.kaggle.com/datasets/piyush86kumar/valorant-vct-2025-all-events) | 1 | 2025 match dates |
| [`piyush86kumar/valorant-champions-tour-2024-all-events`](https://www.kaggle.com/datasets/piyush86kumar/valorant-champions-tour-2024-all-events) | 3 | 2024 match dates |

Which files the project uses, and why: [`data_sources.md`](data_sources.md).

## Match dates

The main dataset has no dates in any of its 21 files, and honest testing needs matches in
order. Every source was scraped from vlr.gg, so they share match IDs, and
[`src/dates.py`](../src/dates.py) joins the dates on them:

| Season | Dates from | Matches dated |
|---|---|---|
| 2024 | The 2024 Kaggle set (a local `vct_2024/` copy is used first if present) | 434 of 434 |
| 2025 | The 2025 Kaggle set | 503 of 503 |
| 2026 | `vlr_dates_vct_2026.csv`, read from vlr.gg by [`fetch_vlr_dates.py`](../fetch_vlr_dates.py) | 342 of 342 |

- **Order by date, never by match ID.** IDs mostly rise over time but run backwards in 182
  places in 2025, and 160 in 2024.
- **Concurrent matches share a timestamp**, so sort by `(match_datetime, match_id)`.
- **Eight 2025 dates had "Today" or "Yesterday" glued on** by the scrape. `dates.py` strips it.
- **The vlr.gg script was checked on 2025 first.** It matched the Kaggle dates on all 504
  matches once vlr.gg's UK times were converted to the Kaggle source's Indian time. It stores
  match IDs and dates only, never scores.

## Matches after the Kaggle data

The Kaggle data ends at Masters London (21 June 2026). `vlr_maps_vct_2026_late.csv` holds the
702 maps played after it, in Stage 2 (all four leagues) and Champions 2026, read from vlr.gg
match pages by [`fetch_vlr_matches.py`](../fetch_vlr_matches.py) in the same columns as
`build_map_table`, plus the patch. Only the app reads it (`live_map_table` in
`src/dataset.py`); the notebooks' tests stay on Kaggle.

- **Checked against Kaggle first**, on Masters London and China and EMEA Stage 1: all 296
  maps matched on every column.
- **Team names come from each map's header**, which uses Kaggle's short names ("Bilibili
  Gaming", not the sponsor name at the top of the page). The one exception, JD Gaming, is
  mapped to Kaggle's "JDG Esports".
- **14 teams are new to 2026's data.** They're Challengers teams from the Stage 2 Play-Ins,
  not misspellings.
- **A match page's `data-utc-ts` is US Eastern time, not UTC.** It's converted to Indian time,
  like every other date in the project.
- The pages are saved in `vlr_pages/` (not committed), so re-running it reads them again
  instead of asking vlr.gg.

## Column reference

The main dataset's Kaggle page describes every file and column. A few column names differ
slightly from the files, so check the headers.
