"""Fetches whole matches from vlr.gg, for events the Kaggle dataset doesn't have yet.

The main dataset stops at Masters London (June 2026). Stage 2 and Champions 2026 were
played after it, with a new map (Summit) and a different meta, so the app needs them.
Every vlr.gg match page shows, per map: both teams' five agents, the score and who
picked the map. That is everything `dataset.build_map_table` makes, so this writes the
same columns, one row per map, to data/vlr_maps_<name>.csv.

Pages are read a few seconds apart, and each is kept in data/vlr_pages/ (not committed),
so a second run re-reads the saved pages instead of asking vlr.gg again. vlr.gg's
robots.txt allows these pages (it disallows only /search/auto and /rr/).

Before being trusted it is run on an event the Kaggle dataset already has, and every
map compared with Kaggle's (`--check`):

    python fetch_vlr_matches.py check 2765 --check     # Masters London 2026
    python fetch_vlr_matches.py vct_2026_late 2976 2977 2978 2776 2766

The first argument names the output file; the rest are vlr.gg event IDs, from the
event's URL (vlr.gg/event/2766/valorant-champions-2026).
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (personal research project)"}
PAUSE_SECONDS = 3
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data"
PAGE_DIR = OUT_DIR / "vlr_pages"

# A match page's `data-utc-ts` is not UTC: checked on Champions 2026, a stamp of 05:00
# sits beside a header reading 10:00 BST, so it is US Eastern time. Times are stored in
# Indian time to match the rest of the project (see fetch_vlr_dates.py).
PAGE_ZONE = "America/New_York"
SOURCE_ZONE = "Asia/Kolkata"

# vlr.gg's short team names that the Kaggle dataset spells differently. Found by `--check`
# on China and EMEA Stage 1; `main` lists any name Kaggle has never used, to add here.
KAGGLE_NAMES = {"JD Gaming": "JDG Esports"}

COLUMNS = ["match_id", "game_id", "played_at", "event", "tournament", "stage", "match_type",
           "match_name", "map", "team_a", "team_b", "comp_a", "comp_b", "map_picked_by",
           "score_a", "score_b", "team_a_won", "patch"]


def page(path: str) -> str:
    """One vlr.gg page, from the saved copy if there is one."""
    saved = PAGE_DIR / (path.strip("/").replace("/", "_").replace("?", "_") + ".html")
    if saved.exists():
        return saved.read_text(encoding="utf-8")
    time.sleep(PAUSE_SECONDS)
    response = requests.get(f"https://www.vlr.gg{path}", headers=HEADERS, timeout=30)
    response.raise_for_status()
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    saved.write_text(response.text, encoding="utf-8")
    return response.text


def event_match_ids(event_id: int) -> list[int]:
    """The IDs of every match listed on one event's match page."""
    soup = BeautifulSoup(page(f"/event/matches/{event_id}/?series_id=all"), "html.parser")
    ids = [int(a["href"].strip("/").split("/")[0]) for a in soup.select("a.match-item")]
    return list(dict.fromkeys(ids))


def agent_name(title: str) -> str:
    """vlr.gg's agent name in this project's spelling: "KAY/O" -> "kayo"."""
    return re.sub(r"[^a-z]", "", title.lower())


def match_maps(match_id: int) -> list[dict]:
    """One row per map played in one match. Maps that weren't played are left out."""
    soup = BeautifulSoup(page(f"/{match_id}"), "html.parser")
    # The page's main header gives sponsor names ("JD Mall JDG Esports (JD Gaming)"); each
    # map's header gives the short name the Kaggle dataset uses, with one exception.
    team_1, team_2 = (KAGGLE_NAMES.get(n, n) for n in
                      (t.get_text(strip=True) for t in soup.select(".vm-stats-game-header .team-name")[:2]))

    header = soup.select_one(".match-header-super")
    # "VCT 2026: EMEA Stage 2" -> "VCT 2026 EMEA Stage 2", as the Kaggle dataset writes it
    event = header.select_one(".match-header-event div div").get_text(" ", strip=True).replace(":", "")
    series = header.select_one(".match-header-event-series").get_text(" ", strip=True)
    stage, _, match_type = (part.strip() for part in series.partition(":"))
    stamp = header.select_one(".moment-tz-convert[data-utc-ts]")["data-utc-ts"]
    played_at = (pd.Timestamp(stamp).tz_localize(PAGE_ZONE)
                 .tz_convert(SOURCE_ZONE).tz_localize(None))
    patch = re.search(r"Patch\s+([\d.]+)", header.get_text(" ", strip=True))

    rows = []
    for game in soup.select("div.vm-stats-game"):
        if game.get("data-game-id") == "all":
            continue
        name = game.select_one(".map-name > span")
        if name is None:
            continue
        picked = name.select_one(".picked")
        map_name = name.find(string=True, recursive=False).strip()
        scores = [int(s.get_text(strip=True)) for s in game.select(".vm-stats-game-header .score")]
        tables = game.select("div.ovw-table")  # one per team, team 1 first
        comps = [tuple(sorted(agent_name(img["title"]) for img in t.select(".mod-agents img[title]")))
                 for t in tables]
        if len(comps) != 2 or any(len(c) != 5 for c in comps) or len(scores) != 2 or not any(scores):
            continue  # listed but not played (e.g. the third map of a 2-0)

        classes = picked.get("class", []) if picked else []
        picked_by = "team_a" if "mod-1" in classes else "team_b" if "mod-2" in classes else "decider"
        rows.append({
            "match_id": match_id, "game_id": int(game["data-game-id"]),
            "played_at": played_at, "event": event, "tournament": event,
            "stage": stage, "match_type": match_type, "match_name": f"{team_1} vs {team_2}",
            "map": map_name, "team_a": team_1, "team_b": team_2,
            "comp_a": comps[0], "comp_b": comps[1], "map_picked_by": picked_by,
            "score_a": scores[0], "score_b": scores[1], "team_a_won": int(scores[0] > scores[1]),
            "patch": patch.group(1) if patch else None,
        })
    return rows


def fetch_events(event_ids: list[int]) -> pd.DataFrame:
    rows = []
    for event_id in event_ids:
        ids = event_match_ids(event_id)
        before = len(rows)
        for match_id in ids:
            rows.extend(match_maps(match_id))
        print(f"  event {event_id}: {len(ids)} matches, {len(rows) - before} maps")
    return pd.DataFrame(rows, columns=COLUMNS)


def kaggle_2026() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from dataset import build_map_table
    return build_map_table("vct_2026")


def new_team_names(scraped: pd.DataFrame) -> set[str]:
    """Teams the Kaggle dataset never names in 2026: a new team, or a spelling to map."""
    trusted = kaggle_2026()
    known = set(trusted["team_a"]) | set(trusted["team_b"])
    return (set(scraped["team_a"]) | set(scraped["team_b"])) - known


def check_against_kaggle(scraped: pd.DataFrame) -> None:
    """Compare every map with the Kaggle dataset, which is trusted, on the same events."""
    trusted = kaggle_2026()
    trusted = trusted[trusted["match_id"].isin(scraped["match_id"])]
    both = scraped.merge(trusted, on="game_id", suffixes=("_vlr", "_kaggle"))
    print(f"\n  maps: {len(scraped)} from vlr.gg, {len(trusted)} in Kaggle, {len(both)} matched")
    for column in ["match_id", "played_at", "map", "team_a", "team_b", "comp_a", "comp_b",
                   "map_picked_by", "score_a", "score_b", "team_a_won"]:
        same = both[f"{column}_vlr"] == both[f"{column}_kaggle"]
        print(f"  {column:14} same on {same.mean():6.1%}")
        if not same.all():
            print(both.loc[~same, [f"{column}_vlr", f"{column}_kaggle"]].head(5).to_string())


def main() -> None:
    name, *rest = sys.argv[1:]
    event_ids = [int(a) for a in rest if a.isdigit()]
    print(f"fetching events {event_ids} from vlr.gg")
    scraped = fetch_events(event_ids)
    path = OUT_DIR / f"vlr_maps_{name}.csv"
    scraped.to_csv(path, index=False)
    print(f"\n  {len(scraped)} maps from {scraped['match_id'].nunique()} matches, "
          f"{scraped['played_at'].min():%d %b %Y} to {scraped['played_at'].max():%d %b %Y} "
          f"-> {path.name}")
    print(f"  maps: {scraped['map'].value_counts().to_dict()}")
    print(f"  teams Kaggle never names in 2026: {sorted(new_team_names(scraped)) or 'none'}")
    if "--check" in rest:
        check_against_kaggle(scraped)


if __name__ == "__main__":
    main()
