"""Match dates for a VCT season.

The VCT 2021-2026 dataset (our primary source) has no date column in any of its
21 tables, so matches can't be ordered in time -- which a chronological
train/test split depends on. The VCT 2025 All Events dataset does have dates,
and both are scraped from vlr.gg, so they share a match ID space:
`match_id` there == `Match ID` in the primary dataset's ids table.

This builds that bridge. Verified against vct_2025: all 503 primary-dataset
matches get a date, every tournament at 100%. Verified against vct_2024: all 434
get a date, every tournament at 100%.

Both date sources come from the same author and share one layout -- a folder per
event, each with a matches.csv holding the date and the local start time.
"""

import glob
import os
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "piyush86kumar/valorant-vct-2025-all-events"
DATASET_2024 = "piyush86kumar/valorant-champions-tour-2024-all-events"

# Where each season's dates come from. 2025 and 2024 download through kagglehub like
# the rest of the project. A `vct_2024/` folder at the repo root (gitignored, as all
# raw data is) is used instead if present: it is the same dataset, checked file for
# file, and it is where the 2024 dates were first read from.
_LOCAL_2024 = Path(__file__).resolve().parent.parent / "vct_2024"

# 2026 has no dated dataset anywhere, so its dates were read from vlr.gg directly by
# fetch_vlr_dates.py -- checked first on 2025, where it matched the Kaggle source on
# every one of 504 matches, date and time. Only match IDs and dates are stored.
_VLR_DATES = Path(__file__).resolve().parent.parent / "data"
_FROM_VLR = {"vct_2026"}

# vlr.gg renders a "Today"/"Yesterday" badge next to recent matches; the scrape
# concatenated it onto the date string for 8 rows (e.g. "Sun, October 5, 2025Today").
_BADGE = r"(Today|Yesterday|Tomorrow)\s*$"

_DATE_FMT = "%a, %B %d, %Y"
_DATETIME_FMT = "%a, %B %d, %Y %I:%M %p"


def _date_source(season: str) -> str:
    """The folder holding one sub-folder per event for this season."""
    if season == "vct_2025":
        return kagglehub.dataset_download(DATASET)
    if season == "vct_2024":
        if _LOCAL_2024.is_dir():
            return str(_LOCAL_2024)
        return kagglehub.dataset_download(DATASET_2024)
    msg = f"No date source for {season}."
    raise ValueError(msg)


def load_match_dates(season: str = "vct_2025") -> pd.DataFrame:
    """Return one row per match: match_id, match_date, match_datetime, event.

    Sort by (match_datetime, match_id) for a stable chronological order --
    concurrent matches share a timestamp, so match_id breaks those ties.
    """
    if season in _FROM_VLR:
        dates = pd.read_csv(_VLR_DATES / f"vlr_dates_{season}.csv",
                            parse_dates=["match_date", "match_datetime"])
        out = dates.assign(stage=pd.NA, week=pd.NA)[
            ["match_id", "match_date", "match_datetime", "event", "stage", "week"]]
        return out.sort_values(["match_datetime", "match_id"]).reset_index(drop=True)

    root = _date_source(season)
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
