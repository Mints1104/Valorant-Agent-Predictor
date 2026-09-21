"""Fetches match dates from vlr.gg for a season, for seasons no dataset dates.

The main dataset has no dates at all. For 2024 and 2025 they come from other
Kaggle datasets scraped from vlr.gg; for 2026 no such dataset exists, so this
reads them from vlr.gg directly, the same way those datasets were built.

It keeps **only the match ID, date and start time** of each match. vlr.gg's
match lists also show scores; this script never reads them, so it cannot leak a
sealed season's results (2026 is sealed -- see CLAUDE.md).

**Time zones.** vlr.gg shows times in the visitor's own zone -- UK time, from here --
while the Kaggle date sources for 2024 and 2025 are in Indian time (UTC+5:30). Checked on
2025, the gap was exactly 5h30 in winter and 4h30 in British summer time, so times are
converted from Europe/London to Asia/Kolkata to match. Run it from somewhere else and
LOCAL_ZONE must change.

One page per event, a few seconds apart. vlr.gg's robots.txt allows these pages
(it disallows only /search/auto and /rr/). The result is saved to
data/vlr_dates_<season>.csv so this only ever needs running once.

    python fetch_vlr_dates.py vct_2026

Before being trusted for 2026 it was run on vct_2025 and checked against the
Kaggle date source: see the output of `--check`.

    python fetch_vlr_dates.py vct_2025 --check
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import load_primary  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (personal research project)"}
PAUSE_SECONDS = 3
OUT_DIR = Path(__file__).parent / "data"

LOCAL_ZONE = "Europe/London"      # the zone vlr.gg displays for this machine
SOURCE_ZONE = "Asia/Kolkata"      # the zone the Kaggle date sources use

_BADGE = re.compile(r"\s*(Today|Yesterday|Tomorrow)\s*$")


def event_matches(event_id: int) -> list[dict]:
    """Every match on one vlr.gg event page: its ID, date and start time. Nothing else."""
    url = f"https://www.vlr.gg/event/matches/{event_id}/?series_id=all"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    rows = []
    for label in soup.select("div.wf-label.mod-large"):
        day = _BADGE.sub("", label.get_text(" ", strip=True))
        card = label.find_next_sibling("div", class_="wf-card")
        if card is None:
            continue
        for item in card.select("a.match-item"):
            match_id = int(item["href"].strip("/").split("/")[0])
            clock = item.select_one(".match-item-time")
            rows.append({"match_id": match_id, "date": day,
                         "time": clock.get_text(strip=True) if clock else ""})
    return rows


def fetch_season(season: str) -> pd.DataFrame:
    ids = load_primary("ids/tournaments_stages_matches_games_ids.csv", season)
    events = ids[["Tournament", "Tournament ID"]].drop_duplicates()
    frames = []
    for n, (name, event_id) in enumerate(events.itertuples(index=False)):
        if n:
            time.sleep(PAUSE_SECONDS)
        rows = event_matches(int(event_id))
        print(f"  {name:34} {len(rows):3} matches")
        frames.append(pd.DataFrame(rows).assign(event=name.replace(":", "")))
    out = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")

    out["match_date"] = pd.to_datetime(out["date"], format="%a, %B %d, %Y")
    clock = out["time"].where(out["time"].str.match(r"^\d{1,2}:\d{2} [AP]M$"))
    local = pd.to_datetime(out["date"] + " " + clock.fillna("12:00 AM"), format="%a, %B %d, %Y %I:%M %p")
    out["match_datetime"] = (local.dt.tz_localize(LOCAL_ZONE)
                             .dt.tz_convert(SOURCE_ZONE).dt.tz_localize(None))
    out["match_date"] = out["match_datetime"].dt.normalize()
    out["time_known"] = clock.notna()
    return out[["match_id", "match_date", "match_datetime", "time_known", "event"]]


def check_against_kaggle(scraped: pd.DataFrame, season: str) -> None:
    """Compare with the Kaggle date source, which is already trusted for this season."""
    from dates import load_match_dates

    trusted = load_match_dates(season)[["match_id", "match_date", "match_datetime"]]
    both = scraped.merge(trusted, on="match_id", suffixes=("_vlr", "_kaggle"))
    same_day = (both["match_date_vlr"] == both["match_date_kaggle"]).mean()
    timed = both[both["time_known"]]
    same_time = (timed["match_datetime_vlr"] == timed["match_datetime_kaggle"]).mean()
    order = both.sort_values(["match_datetime_kaggle", "match_id"])["match_id"].to_numpy()
    order_vlr = both.sort_values(["match_datetime_vlr", "match_id"])["match_id"].to_numpy()
    print(f"\n  matched to the Kaggle date source: {len(both)} of {len(trusted)} matches")
    print(f"  same date: {same_day:.1%}   same date and time, where vlr.gg gives a time "
          f"({len(timed)} matches): {same_time:.1%}")
    print(f"  same order of matches: {(order == order_vlr).mean():.1%}")
    off = both[both["match_date_vlr"] != both["match_date_kaggle"]]
    if len(off):
        print((off["match_date_vlr"] - off["match_date_kaggle"]).value_counts().to_string())


def main() -> None:
    season = sys.argv[1]
    print(f"fetching {season} from vlr.gg")
    scraped = fetch_season(season)
    path = OUT_DIR / f"vlr_dates_{season}.csv"
    scraped.to_csv(path, index=False)
    print(f"\n  {len(scraped)} matches, {scraped['match_date'].min():%d %b %Y} to "
          f"{scraped['match_date'].max():%d %b %Y} -> {path.name}")
    if "--check" in sys.argv:
        check_against_kaggle(scraped, season)


if __name__ == "__main__":
    main()
