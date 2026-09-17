"""Turns each map into numbers a model can read.

A team's line-up is five agent names bundled together. There are 459 different
bundles in the 2025 season and a third of them were used only once, so treating
each bundle as its own separate thing gives a model almost nothing to learn
from -- it would never see most of them more than once.

Instead each line-up is described by which agents are in it. Two line-ups that
share four agents then look nearly identical to the model, rather than looking
like two unrelated things.

Team A and team B is only the order the two teams happen to be listed in, so
the numbers are written from team A's point of view as a comparison:

    +1  only team A played this agent
    -1  only team B played this agent
     0  both teams played them, or neither did

Agents both teams played cancel out to zero on purpose. They cannot explain why
one side beat the other. Teams share about three of their five agents on an
average map, so this happens often.
"""

import pandas as pd

PICKED_BY_TEAM_A = {"team_a": 1, "team_b": -1, "decider": 0}


def agent_list(maps: pd.DataFrame) -> list[str]:
    """Every agent played during the season, in alphabetical order."""
    played = {agent for comp in maps["comp_a"] for agent in comp}
    played |= {agent for comp in maps["comp_b"] for agent in comp}
    return sorted(played)


def feature_columns(maps: pd.DataFrame) -> list[str]:
    """Every column build_features can produce for this data, in a fixed order.

    Work this out once from the WHOLE season and pass it to every call, or the
    two halves of the season come back different shapes: the map pool rotated
    mid-2025, so Corrode appears only after the split and Fracture, Pearl and
    Split only before it. A model fitted on one half then cannot score the
    other.
    """
    return (
        [f"agent_{agent}" for agent in agent_list(maps)]
        + [f"map_{name}" for name in sorted(maps["map"].unique())]
        + ["map_picked_by_a"]
    )


def build_features(
    maps: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the numbers describing each map, and who won it.

    Everything here is known before a single round is played, which is the whole
    point -- a model built on what happened during the map could not be used to
    predict anything.

    `columns` fixes which columns come back. Pass feature_columns(all_maps) when
    working with part of a season; anything missing from this slice comes back
    as zero, which is the truthful value -- that map was not played here.
    Leaving it as None derives the columns from `maps` alone, which is only safe
    when `maps` is the whole season.
    """
    features = {}

    for agent in agent_list(maps):
        in_a = maps["comp_a"].map(lambda comp, a=agent: a in comp).astype(int)
        in_b = maps["comp_b"].map(lambda comp, a=agent: a in comp).astype(int)
        features[f"agent_{agent}"] = in_a - in_b

    # Which map was played. Each map gets its own column holding 1 or 0, because
    # the maps have no natural order -- Ascent is not "less than" Bind.
    for map_name in sorted(maps["map"].unique()):
        features[f"map_{map_name}"] = (maps["map"] == map_name).astype(int)

    features["map_picked_by_a"] = maps["map_picked_by"].map(PICKED_BY_TEAM_A)

    x = pd.DataFrame(features, index=maps.index)

    if columns is not None:
        unexpected = set(x.columns) - set(columns)
        if unexpected:
            msg = f"These are in the data but not in the column list given: {sorted(unexpected)}"
            raise ValueError(msg)
        x = x.reindex(columns=columns, fill_value=0)

    y = maps["team_a_won"]
    return x, y


def swap_teams(maps: pd.DataFrame) -> pd.DataFrame:
    """Return the same maps with the two teams listed the other way round.

    Used to check the features behave sensibly: since which team is called A is
    arbitrary, swapping the teams should flip every comparison and flip who won.
    """
    swapped = maps.copy()
    swapped[["team_a", "team_b"]] = maps[["team_b", "team_a"]].to_numpy()
    swapped[["comp_a", "comp_b"]] = maps[["comp_b", "comp_a"]].to_numpy()
    swapped[["score_a", "score_b"]] = maps[["score_b", "score_a"]].to_numpy()
    swapped["map_picked_by"] = maps["map_picked_by"].map(
        {"team_a": "team_b", "team_b": "team_a", "decider": "decider"}
    )
    swapped["team_a_won"] = 1 - maps["team_a_won"]
    return swapped
