"""Match dates for a VCT season.

The main dataset has no dates, so matches can't be put in order. Everything here was
scraped from vlr.gg, so match IDs are shared and dates can be joined on them:

- 2024 and 2025: two Kaggle datasets by one author (a folder per event, each with a
  matches.csv). Every match in the main dataset gets a date: 434 in 2024, 503 in 2025.
- 2026: no dataset has dates, so fetch_vlr_dates.py read them from vlr.gg into
  data/vlr_dates_vct_2026.csv. Match IDs and dates only.
"""

import glob
import os
from pathlib import Path

import kagglehub
import pandas as pd

# Pinned versions, for the same reasons as PRIMARY in dataset.py.
DATASET = "piyush86kumar/valorant-vct-2025-all-events/versions/1"
DATASET_2024 = "piyush86kumar/valorant-champions-tour-2024-all-events/versions/3"

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
