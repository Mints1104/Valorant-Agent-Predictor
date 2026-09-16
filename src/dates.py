"""Match dates for the VCT 2025 season.

The VCT 2021-2026 dataset (our primary source) has no date column in any of its
21 tables, so matches can't be ordered in time -- which a chronological
train/test split depends on. The VCT 2025 All Events dataset does have dates,
and both are scraped from vlr.gg, so they share a match ID space:
`match_id` there == `Match ID` in the primary dataset's ids table.

This builds that bridge. Verified against vct_2025: all 503 primary-dataset
matches get a date, every tournament at 100%.
"""

import glob
import os

import kagglehub
import pandas as pd

DATASET = "piyush86kumar/valorant-vct-2025-all-events"

# vlr.gg renders a "Today"/"Yesterday" badge next to recent matches; the scrape
# concatenated it onto the date string for 8 rows (e.g. "Sun, October 5, 2025Today").
_BADGE = r"(Today|Yesterday|Tomorrow)\s*$"

_DATE_FMT = "%a, %B %d, %Y"
_DATETIME_FMT = "%a, %B %d, %Y %I:%M %p"


def load_match_dates() -> pd.DataFrame:
    """Return one row per match: match_id, match_date, match_datetime, event.

    Sort by (match_datetime, match_id) for a stable chronological order --
    concurrent matches share a timestamp, so match_id breaks those ties.
    """
    root = kagglehub.dataset_download(DATASET)
    files = sorted(glob.glob(os.path.join(root, "*", "matches.csv")))
    if not files:
        msg = f"No matches.csv found under {root}"
        raise FileNotFoundError(msg)

    frames = []
    for path in files:
        event = os.path.basename(os.path.dirname(path)).removesuffix("_csvs")
        frames.append(pd.read_csv(path).assign(event=event))
    df = pd.concat(frames, ignore_index=True)

    date_str = df["date"].str.replace(_BADGE, "", regex=True).str.strip()
    df["match_date"] = pd.to_datetime(date_str, format=_DATE_FMT)
    df["match_datetime"] = pd.to_datetime(
        date_str + " " + df["time"].str.strip(), format=_DATETIME_FMT
    )

    out = df[["match_id", "match_date", "match_datetime", "event", "stage", "week"]]
    return out.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)
