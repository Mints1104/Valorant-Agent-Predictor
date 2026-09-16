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


def build_features(maps: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the numbers describing each map, and who won it.

    Everything here is known before a single round is played, which is the whole
    point -- a model built on what happened during the map could not be used to
    predict anything.
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
