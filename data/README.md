# The data

Nothing here is committed. The raw files download through `kagglehub` and are cached on your
machine. The versions are pinned, so after the first download everything works offline.

## Sign in to Kaggle (once)

Create a token at [kaggle.com/settings/api](https://www.kaggle.com/settings/api) ("Generate
New Token") and save it, on its own, in `~/.kaggle/access_token` (on Windows,
`C:\Users\<you>\.kaggle\access_token`). A `.env` file or a `KAGGLE_API_KEY` variable does
nothing; the variable kagglehub reads is `KAGGLE_API_TOKEN`. Then run
`python test_data_pull.py` to check the downloads work.

## The datasets

| Dataset | Version | Gives us |
|---|---|---|
| [`ryanluong1/valorant-champion-tour-2021-2023-data`](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) | 47 | Results, line-ups, map picks and player stats for 2021–2026 (the name was never updated). No dates |
| [`piyush86kumar/valorant-vct-2025-all-events`](https://www.kaggle.com/datasets/piyush86kumar/valorant-vct-2025-all-events) | 1 | 2025 match dates |
| [`piyush86kumar/valorant-champions-tour-2024-all-events`](https://www.kaggle.com/datasets/piyush86kumar/valorant-champions-tour-2024-all-events) | 3 | 2024 match dates |

What each file in the main dataset holds: [`VCT_2025_DATA_SUMMARY.md`](VCT_2025_DATA_SUMMARY.md).
Which ones the project uses, and why: [`data_sources.md`](data_sources.md).

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

## Column reference

[`columns_description.csv`](../columns_description.csv) is the dataset's own field dictionary.
A few column names differ slightly from the files, so check the headers.
