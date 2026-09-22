"""Builds the main table: one row per map played, with both teams' agent picks,
who chose the map, and who won.

This is the table the models are trained on. It applies the clean-up rules
worked out in notebooks/01_eda.ipynb, all of which are needed for correct
results rather than being tidying-up:

  1. overview.csv mixes whole-match summary rows ("All Maps") in with the
     real per-map rows, so those are dropped.
  2. The same file repeats every player three times (whole map, attacking
     half, defending half), so only the whole-map rows are kept.
  3. NRG is recorded under the name "Mega Minors" in the team columns, so
     that name is corrected before anything is matched up by team name.
  4. Five exhibition matches (all-star games and the like) are removed.
"""

import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

from dates import load_match_dates

# Pinned to the version every result was computed on. Unpinned, kagglehub asks Kaggle for
# the latest version on every load: that fails with no internet, and a new upload would
# silently change the numbers.
PRIMARY = "ryanluong1/valorant-champion-tour-2021-2023-data/versions/47"
SEASON = "vct_2025"

# The four columns that together identify one match in this dataset.
MATCH_KEY = ["Tournament", "Stage", "Match Type", "Match Name"]

# NRG appears under an old/alternate name in the team columns only.
TEAM_ALIASES = {"Mega Minors": "NRG"}

# Where the season is cut in two. Everything before this date is learned from,
# everything on or after it is held back for the final score.
#
# The date itself is arbitrary within a range: Masters Toronto finished on the
# night of 22 June and the next tournament started on 4 July, so any date in
# that gap gives the same 756 / 516 split. A date in the middle of the gap is
# used rather than one right next to a match, so that it cannot be misread as
# an off-by-one -- no tournament is ever cut in half.
SPLIT_DATE = pd.Timestamp("2025-06-30")

OUTPUT_COLUMNS = {
    "Match ID": "match_id",
    "Game ID": "game_id",
    "match_datetime": "played_at",
    "event": "event",
    "Tournament": "tournament",
    "Stage": "stage",
    "Match Type": "match_type",
    "Match Name": "match_name",
    "Map": "map",
    "Team A": "team_a",
    "Team B": "team_b",
    "comp_a": "comp_a",
    "comp_b": "comp_b",
    "map_picked_by": "map_picked_by",
    "Team A Score": "score_a",
    "Team B Score": "score_b",
    "team_a_won": "team_a_won",
}


def load_primary(path: str, season: str = SEASON) -> pd.DataFrame:
    """Load one CSV from the main dataset. Downloads once, then reads from cache."""
    return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, PRIMARY, f"{season}/{path}")


def _team_compositions(overview: pd.DataFrame) -> pd.DataFrame:
    """The five agents each team played, per map."""
    per_map = overview["Map"].str.strip().str.lower() != "all maps"
    whole_map_only = overview["Side"] == "both"

    cleaned = overview[per_map & whole_map_only].copy()
    cleaned["Team"] = cleaned["Team"].replace(TEAM_ALIASES)

    return (
        cleaned.groupby(MATCH_KEY + ["Map", "Team"])["Agents"]
        .apply(lambda agents: tuple(sorted(agents)))
        .rename("comp")
        .reset_index()
    )


def build_map_table(season: str = SEASON) -> pd.DataFrame:
    """One row per map played in one VCT season, ready for feature building.

    Defaults to 2025, the season the frozen model was built on. Raises if any
    step loses or duplicates rows, since a silent change in row count almost
    always means a name mismatch rather than a real gap.
    """
    maps_scores = load_primary("matches/maps_scores.csv", season)
    scores = load_primary("matches/scores.csv", season)
    ids = load_primary("ids/tournaments_stages_matches_games_ids.csv", season)
    overview = load_primary("matches/overview.csv", season)
    draft = load_primary("matches/draft_phase.csv", season)

    # Exhibition matches are labelled as such and have made-up line-ups.
    exhibitions = set(
        map(tuple, scores[scores["Stage"] == "Showmatch"][MATCH_KEY].drop_duplicates().values)
    )
    table = maps_scores[
        ~maps_scores.apply(lambda row: tuple(row[MATCH_KEY]) in exhibitions, axis=1)
    ].copy()

    expected_rows = len(table)

    for column in ("Team A", "Team B"):
        table[column] = table[column].replace(TEAM_ALIASES)

    table = table.merge(ids[MATCH_KEY + ["Map", "Match ID", "Game ID"]],
                        on=MATCH_KEY + ["Map"], how="left")

    dates = load_match_dates(season)
    table = table.merge(dates[["match_id", "match_datetime", "event"]],
                        left_on="Match ID", right_on="match_id", how="left")

    comps = _team_compositions(overview)
    table = table.merge(comps.rename(columns={"Team": "Team A", "comp": "comp_a"}),
                        on=MATCH_KEY + ["Map", "Team A"], how="left")
    table = table.merge(comps.rename(columns={"Team": "Team B", "comp": "comp_b"}),
                        on=MATCH_KEY + ["Map", "Team B"], how="left")

    # Who chose this map. A match's final map is whatever is left after both
    # teams have picked, so nobody picks it -- that is a real category here,
    # not a gap in the data.
    draft = draft.copy()
    draft["Team"] = draft["Team"].replace(TEAM_ALIASES)
    picks = (
        draft[draft["Action"] == "pick"][MATCH_KEY + ["Map", "Team"]]
        .rename(columns={"Team": "picked_by"})
    )
    table = table.merge(picks, on=MATCH_KEY + ["Map"], how="left")

    table["map_picked_by"] = "decider"
    table.loc[table["picked_by"] == table["Team A"], "map_picked_by"] = "team_a"
    table.loc[table["picked_by"] == table["Team B"], "map_picked_by"] = "team_b"

    table["team_a_won"] = (table["Team A Score"] > table["Team B Score"]).astype(int)

    if len(table) != expected_rows:
        msg = f"Row count changed while joining: {expected_rows} -> {len(table)}"
        raise ValueError(msg)

    missing = table[["Match ID", "match_datetime", "comp_a", "comp_b"]].isna().sum()
    if missing.any():
        msg = f"Unexpected gaps after joining:\n{missing[missing > 0]}"
        raise ValueError(msg)

    return (
        table[list(OUTPUT_COLUMNS)]
        .rename(columns=OUTPUT_COLUMNS)
        .sort_values(["played_at", "match_id", "game_id"])
        .reset_index(drop=True)
    )


def split_season(maps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the season into the part we learn from and the part we are scored on.

    Returns (learn_from, test_on). Defined here so every notebook cuts the
    season in the same place -- the date used to be written out by hand in each
    one, which is exactly how they drift apart.
    """
    earlier = maps["played_at"] < SPLIT_DATE
    return maps[earlier].copy(), maps[~earlier].copy()


N_FOLDS = 5


def season_folds(maps: pd.DataFrame, n_folds: int = N_FOLDS):
    """Split maps into growing train / next-chunk validation pairs, oldest first.

    Cuts between matches rather than between maps, so no match ever has some of
    its maps used for training and the rest for checking -- whichever team is
    playing better on the night tends to take several maps, so a match split
    across the line would flatter the score.

    Yields pairs of row labels usable with .loc. Defined here rather than in a
    notebook because more than one notebook needs it, and hand-copied logic is
    how the split date drifted the first time.
    """
    in_order = maps.sort_values(["played_at", "match_id"])
    matches = in_order["match_id"].drop_duplicates().to_numpy()
    chunks = np.array_split(matches, n_folds + 1)

    for fold in range(n_folds):
        learn_ids = np.concatenate(chunks[: fold + 1])
        check_ids = chunks[fold + 1]
        yield (
            maps.index[maps["match_id"].isin(learn_ids)],
            maps.index[maps["match_id"].isin(check_ids)],
        )
