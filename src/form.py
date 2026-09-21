"""Turns how the ten players have been performing into numbers for the map ahead.

This is the "with player stats" half of the project plan. The rule it has to
obey is that a map's numbers may only use matches played *earlier* than the one
being predicted. Break that and the model is reading the result off the back of
the page: it scores brilliantly and could never be used for anything, because in
real life nobody has played the match yet.

The numbers are built by walking through the season in date order. For each
match, each side's form is worked out from what is already recorded, and only
then does that match get added to the players' records. Because the second step
comes after the first, a match can never contribute to its own numbers. Working
per match rather than per map also stops map 1 of a match informing map 3 of the
same match, which matters because whichever team is playing better on the night
tends to take several maps.

A player's recent form can matter more than their form in January, so older
maps can be given less weight -- see `half_life`.

One gap to know about: `Rating`, `Average Damage Per Round` and
`Kill, Assist, Trade, Survive %` are missing for all of China Kickoff and much
of China Stage 1. `Average Combat Score` is very nearly complete, which makes it
the safer thing to build on. See data/data_sources.md.
"""

import numpy as np
import pandas as pd

from dataset import TEAM_ALIASES, load_primary

# The per-map performance columns worth using, and the short names used for the
# features built from them.
STATS = {
    "Rating": "rating",
    "Average Combat Score": "acs",
    "Average Damage Per Round": "adr",
    "Kill, Assist, Trade, Survive %": "kast",
}

MATCH_KEY = ["tournament", "stage", "match_type", "match_name", "map"]

_RENAME = {
    "Tournament": "tournament",
    "Stage": "stage",
    "Match Type": "match_type",
    "Match Name": "match_name",
    "Map": "map",
}


def load_player_maps(maps: pd.DataFrame) -> pd.DataFrame:
    """One row per player per map, limited to the maps given.

    Applies the same two filters as everywhere else -- drop the "All Maps"
    summary rows, keep only whole-map rows -- without which every player is
    counted several times over. Raises if the row count is not ten per map,
    since that means a join has quietly lost somebody.
    """
    # Tournament names carry the year ("VCT 2025: ...", "Champions Tour 2024: ..."),
    # so rows from different seasons can never be matched to the wrong map.
    seasons = sorted({f"vct_{year}" for year in maps["played_at"].dt.year})
    overview = pd.concat([load_primary("matches/overview.csv", s) for s in seasons],
                         ignore_index=True)

    per_map = overview["Map"].str.strip().str.lower() != "all maps"
    whole_map = overview["Side"] == "both"

    players = overview[per_map & whole_map].copy()
    players["Team"] = players["Team"].replace(TEAM_ALIASES)
    players = players.rename(columns=_RENAME)

    for column in STATS:
        if not pd.api.types.is_numeric_dtype(players[column]):
            players[column] = pd.to_numeric(
                players[column].astype(str).str.rstrip("%"), errors="coerce"
            )

    players = players.merge(
        maps[MATCH_KEY + ["match_id", "played_at", "team_a", "team_b"]],
        on=MATCH_KEY,
        how="inner",
    )

    expected = len(maps) * 10
    if len(players) != expected:
        msg = (
            f"Expected 10 player rows per map ({expected:,}), got {len(players):,}. "
            "A team name that does not match, or a map missing from overview.csv."
        )
        raise ValueError(msg)

    return players


def _decay_for(half_life: float | None) -> float:
    """How much each existing performance is faded when a new one arrives.

    `half_life` is counted in maps: with 20, a performance from 20 maps ago
    counts half as much as today's. None means no fading at all -- every map a
    player has ever played counts the same, which is what the first probe did.
    """
    if half_life is None:
        return 1.0
    if half_life <= 0:
        msg = f"half_life must be positive or None, got {half_life}"
        raise ValueError(msg)
    return float(0.5 ** (1.0 / half_life))


def player_form(
    maps: pd.DataFrame,
    half_life: float | None = None,
    stats: dict[str, str] | None = None,
) -> pd.DataFrame:
    """How the two sides had been performing going into each map.

    Returns one row per map in `maps`, in the same order, holding the difference
    between the sides from team A's point of view -- positive means team A's
    players had the better record. Alongside each stat is `games_diff`, how many
    more maps team A's players had played, because a difference between two
    well-established sides means more than one between two newcomers.

    A side with no history at all comes back as missing rather than as zero. The
    caller decides what to do with that; filling it with zero says "no
    difference between these teams", which is the honest thing to say when
    nothing is known about either.
    """
    stats = stats or STATS
    decay = _decay_for(half_life)

    players = load_player_maps(maps)

    totals: dict[tuple[str, str], float] = {}
    weights: dict[tuple[str, str], float] = {}
    games: dict[str, float] = {}
    rows = []

    ordered = players.sort_values(["played_at", "match_id"])

    for (_, match_id), match in ordered.groupby(["played_at", "match_id"], sort=True):
        team_a = match.iloc[0]["team_a"]
        team_b = match.iloc[0]["team_b"]

        def side_form(team: str) -> dict[str, float]:
            names = match[match["Team"] == team]["Player"].unique()
            out = {}
            for column, short in stats.items():
                known = [
                    totals[(name, column)] / weights[(name, column)]
                    for name in names
                    if weights.get((name, column), 0) > 0
                ]
                out[short] = float(np.mean(known)) if known else np.nan
            played = [games.get(name, 0) for name in names]
            out["games"] = float(np.mean(played)) if played else 0.0
            return out

        # 1. read the record as it stands, before this match is in it
        form_a, form_b = side_form(team_a), side_form(team_b)

        row = {"match_id": match_id, "games_diff": form_a["games"] - form_b["games"]}
        for short in stats.values():
            row[f"{short}_diff"] = form_a[short] - form_b[short]
        rows.append(row)

        # 2. and only now does this match count towards anybody
        for _, performance in match.iterrows():
            name = performance["Player"]
            games[name] = games.get(name, 0) + 1
            for column in stats:
                value = performance[column]
                if pd.isna(value):
                    continue
                key = (name, column)
                totals[key] = totals.get(key, 0.0) * decay + float(value)
                weights[key] = weights.get(key, 0.0) * decay + 1.0

    per_match = pd.DataFrame(rows).set_index("match_id")

    if per_match.index.duplicated().any():
        msg = "A match id appeared twice while building form -- check the grouping."
        raise ValueError(msg)

    form = maps[["match_id"]].join(per_match, on="match_id").drop(columns="match_id")

    if len(form) != len(maps):
        msg = f"Row count changed while attaching form: {len(maps)} -> {len(form)}"
        raise ValueError(msg)

    return form


def current_team_form(
    maps: pd.DataFrame,
    half_life: float | None = None,
    stats: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Each team's form as it stands at the end of the data given.

    Where `player_form` answers "how were these sides playing going into that
    map", this answers "how is each team playing now", which is what a live
    prediction needs. Returns one row per team, averaged across the five
    players who most recently appeared for them, so a team that changed its
    roster is judged on the players it currently fields.

    Only for predicting matches after `maps` ends. Using it on a map inside
    `maps` would be using that map's own result.
    """
    stats = stats or STATS
    decay = _decay_for(half_life)

    players = load_player_maps(maps)
    ordered = players.sort_values(["played_at", "match_id"])

    totals: dict[tuple[str, str], float] = {}
    weights: dict[tuple[str, str], float] = {}
    appearances: dict[str, int] = {}
    latest_roster: dict[str, list[str]] = {}

    for _, performance in ordered.iterrows():
        name = performance["Player"]
        appearances[name] = appearances.get(name, 0) + 1
        for column in stats:
            value = performance[column]
            if pd.isna(value):
                continue
            key = (name, column)
            totals[key] = totals.get(key, 0.0) * decay + float(value)
            weights[key] = weights.get(key, 0.0) * decay + 1.0

    # whoever played the team's most recent map is taken as its current roster
    for team, rows in ordered.groupby("Team"):
        last_map = rows.iloc[-1][["played_at", "match_id", "map"]]
        final = rows[
            (rows["played_at"] == last_map["played_at"])
            & (rows["match_id"] == last_map["match_id"])
            & (rows["map"] == last_map["map"])
        ]
        latest_roster[team] = sorted(final["Player"].unique())

    rows = []
    for team, roster in latest_roster.items():
        entry = {"team": team, "players": ", ".join(roster), "roster_size": len(roster)}
        for column, short in stats.items():
            known = [
                totals[(p, column)] / weights[(p, column)]
                for p in roster
                if weights.get((p, column), 0) > 0
            ]
            entry[short] = float(np.mean(known)) if known else np.nan
        entry["maps_played"] = float(np.mean([appearances.get(p, 0) for p in roster]))
        rows.append(entry)

    return pd.DataFrame(rows).set_index("team").sort_index()
