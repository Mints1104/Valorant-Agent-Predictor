"""Recommends agents the way professional teams actually pick them.

Given a map, and optionally some agents already locked in, rank every other agent
by how likely a professional team would be to play it. This describes what the
pros *do*. It does not claim those picks win more -- the win-prediction work in
this project found that agent picks do not predict the winner -- so any win
evidence is shown beside a recommendation, never mixed into it.

Three ways of ranking, all scored the same way:

- **Popularity** -- the naive rule. How often each agent has been played on this
  map lately. It ignores the agents already locked in.
- **Co-occurrence** -- the same count, but a past line-up gets more say the more
  of the locked-in agents it shares. A line-up sharing all four locked agents
  counts SHARE_BOOST ** 4 times as much as one sharing none, so when enough exact
  matches exist they decide, and when they don't it falls back smoothly on looser
  ones. With nothing locked in, it is the same as popularity.
- **Classifier** -- a model that learns "on this map, with these agents locked
  in, which agent completes the line-up?" from every past line-up with each agent
  held out in turn, against every subset of the rest. By default a logistic
  regression, which *adds up* a map effect and one effect per locked agent. With
  `interactions=True` it also gets a column for every map-and-agent pair, so it
  can learn that the same agents lead to different picks on different maps. Any
  scikit-learn classifier can be passed instead -- a gradient-boosted one learns
  such combinations by itself.

All three favour recent line-ups: a line-up played `half_life` days before the
recommendation counts half as much as one played that day, because the agents
teams prefer shift through a season.

Scoring is walk-forward (`walk_forward`): each week of line-ups is recommended
using only line-ups played before that week, the way the tool would be used for
real. That also lets it pick up new agents and maps once pros start playing them.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

LINEUP_SIZE = 5

# How much more a past line-up counts for each locked-in agent it shares. Fixed by
# design rather than tuned: at 10, one extra shared agent outweighs everything else
# about a line-up except recency.
SHARE_BOOST = 10.0

# A whisper of the all-maps ranking, added under every per-map ranking. It only
# decides anything when a map has no history at all (a brand-new map) or two agents
# are otherwise tied.
_ALL_MAPS_WEIGHT = 1e-6


def lineup_table(maps: pd.DataFrame) -> pd.DataFrame:
    """One row per team per map: when, which map, which team, and its five agents."""
    sides = [
        maps[["played_at", "match_id", "map", "team_a", "comp_a"]]
        .rename(columns={"team_a": "team", "comp_a": "agents"}),
        maps[["played_at", "match_id", "map", "team_b", "comp_b"]]
        .rename(columns={"team_b": "team", "comp_b": "agents"}),
    ]
    out = pd.concat(sides, ignore_index=True)
    out["agents"] = out["agents"].map(tuple)
    return out.sort_values(["played_at", "match_id"]).reset_index(drop=True)


def recency_weights(played_at: pd.Series, as_of: pd.Timestamp, half_life: float | None) -> np.ndarray:
    """How much each past line-up counts. None means every line-up counts the same."""
    if half_life is None:
        return np.ones(len(played_at))
    age_days = (as_of - played_at).dt.total_seconds().to_numpy() / 86_400
    return 0.5 ** (age_days / half_life)


class _Counting:
    """Shared machinery for popularity and co-occurrence: weighted counts of past line-ups."""

    share_boost = 1.0

    def __init__(self, half_life: float | None = None):
        self.half_life = half_life

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp):
        self.agents = sorted({a for lineup in history["agents"] for a in lineup})
        self._index = {a: i for i, a in enumerate(self.agents)}
        matrix = np.zeros((len(history), len(self.agents)))
        for row, lineup in enumerate(history["agents"]):
            matrix[row, [self._index[a] for a in lineup]] = 1.0
        weight = recency_weights(history["played_at"], as_of, self.half_life)

        self._all = (matrix, weight)
        maps = history["map"].to_numpy()
        self._by_map = {m: (matrix[maps == m], weight[maps == m]) for m in np.unique(maps)}
        return self

    def _votes(self, matrix, weight, locked_vector):
        shared = matrix @ locked_vector
        return (weight * self.share_boost ** shared) @ matrix

    def scores(self, map_name: str, locked=()) -> pd.Series:
        """A score for every known agent; higher means more likely to be picked."""
        locked_vector = np.zeros(len(self.agents))
        locked_vector[[self._index[a] for a in locked if a in self._index]] = 1.0
        out = _ALL_MAPS_WEIGHT * self._votes(*self._all, locked_vector)
        if map_name in self._by_map:
            out = out + self._votes(*self._by_map[map_name], locked_vector)
        return pd.Series(out, index=self.agents)


class Popularity(_Counting):
    """The naive rule: the most-played agents on this map lately."""

    share_boost = 1.0


class CoOccurrence(_Counting):
    """Popularity, but past line-ups that share the locked-in agents count far more."""

    share_boost = SHARE_BOOST


class Classifier:
    """Which agent completes this line-up, on this map? Logistic regression by default."""

    def __init__(self, half_life: float | None = None, interactions: bool = False, estimator=None):
        self.half_life = half_life
        self.interactions = interactions
        self.estimator = estimator

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp):
        self.agents = sorted({a for lineup in history["agents"] for a in lineup})
        self.maps = sorted(history["map"].unique())
        weight = recency_weights(history["played_at"], as_of, self.half_life)

        # Every line-up teaches every way it could be completed: each agent held out,
        # against every subset of the other four (none locked in up to all four).
        # Identical rows are merged and their weights added, which keeps this small.
        rows = {}
        for lineup, map_name, w in zip(history["agents"], history["map"], weight):
            for held_out in lineup:
                rest = [a for a in lineup if a != held_out]
                for size in range(LINEUP_SIZE):
                    for locked in combinations(rest, size):
                        key = (map_name, locked, held_out)
                        rows[key] = rows.get(key, 0.0) + w

        keys = list(rows)
        X = np.vstack([self._features(m, locked) for m, locked, _ in keys])
        y = np.array([held for _, _, held in keys])
        sample_weight = np.array([rows[k] for k in keys])
        model = clone(self.estimator) if self.estimator is not None else LogisticRegression(max_iter=3000)
        self.model = model.fit(X, y, sample_weight=sample_weight)
        return self

    def _features(self, map_name: str, locked) -> np.ndarray:
        map_part = np.array([map_name == m for m in self.maps], dtype=float)
        agent_part = np.array([a in locked for a in self.agents], dtype=float)
        parts = [map_part, agent_part]
        if self.interactions:
            parts.append(np.outer(map_part, agent_part).ravel())
        return np.concatenate(parts)

    def scores(self, map_name: str, locked=()) -> pd.Series:
        """Probability of each agent being the next pick; a new map has no map column set."""
        chance = self.model.predict_proba(self._features(map_name, set(locked))[None, :])[0]
        return pd.Series(chance, index=self.model.classes_).reindex(self.agents, fill_value=0.0)


def rank_of(scores: pd.Series, locked, target: str) -> float:
    """Where `target` lands among the agents not yet locked in: 1 is the top pick.

    Ties are counted against the target, so a method never gains from a tie. An
    agent the method has never seen cannot be recommended and counts as a miss.
    """
    if target not in scores.index:
        return np.inf
    candidates = scores.drop([a for a in locked if a in scores.index])
    better_or_level = (candidates >= candidates[target]).sum()   # includes the target itself
    return float(better_or_level)


def build_from_scratch(model, map_name: str) -> list[str]:
    """A whole line-up from the map alone, one pick at a time."""
    picked: list[str] = []
    for _ in range(LINEUP_SIZE):
        scores = model.scores(map_name, picked).drop(picked, errors="ignore")
        picked.append(scores.idxmax())
    return picked


def walk_forward(history: pd.DataFrame, to_score: pd.DataFrame, make_model) -> pd.DataFrame:
    """Recommend each week of `to_score` using only line-ups played before that week.

    Returns one row per line-up scored: the rank of each of its five agents when the
    other four are locked in, and how many of its five agents the method would
    have picked from the map alone.
    """
    everything = pd.concat([history, to_score], ignore_index=True)
    week_start = to_score["played_at"].dt.to_period("W").dt.start_time
    results = []
    for start in sorted(week_start.unique()):
        past = everything[everything["played_at"] < start]
        model = make_model().fit(past, as_of=start)
        for idx in to_score.index[week_start == start]:
            lineup = to_score.at[idx, "agents"]
            map_name = to_score.at[idx, "map"]
            ranks = []
            for target in lineup:
                locked = [a for a in lineup if a != target]
                ranks.append(rank_of(model.scores(map_name, locked), locked, target))
            from_scratch = build_from_scratch(model, map_name)
            results.append({
                "row": idx, "week": start, "map": map_name, "match_id": to_score.at[idx, "match_id"],
                "ranks": ranks, "from_scratch_overlap": len(set(from_scratch) & set(lineup)),
            })
    return pd.DataFrame(results).set_index("row")


def summarise(results: pd.DataFrame) -> dict:
    """Top-1 and top-3 on the complete-a-line-up task, and the from-scratch overlap."""
    ranks = np.concatenate(results["ranks"].to_numpy())
    return {
        "top-1": float((ranks <= 1).mean()),
        "top-3": float((ranks <= 3).mean()),
        "from scratch, of 5": float(results["from_scratch_overlap"].mean()),
    }


def lineup_records(maps: pd.DataFrame) -> pd.DataFrame:
    """One row per team per map, with the opponent and the result -- for showing evidence.

    Kept separate from `lineup_table`, which the recommender learns from and which never
    needs to know who won. Results are only ever shown to a person, never used to rank.
    """
    columns = ["played_at", "match_id", "event", "map"]
    a_side = maps[columns].assign(
        team=maps["team_a"], opponent=maps["team_b"], agents=maps["comp_a"].map(tuple),
        rounds_for=maps["score_a"], rounds_against=maps["score_b"], won=maps["team_a_won"] == 1,
    )
    b_side = maps[columns].assign(
        team=maps["team_b"], opponent=maps["team_a"], agents=maps["comp_b"].map(tuple),
        rounds_for=maps["score_b"], rounds_against=maps["score_a"], won=maps["team_a_won"] == 0,
    )
    out = pd.concat([a_side, b_side], ignore_index=True)
    return out.sort_values(["played_at", "match_id"], ascending=False).reset_index(drop=True)


def games_with(records: pd.DataFrame, map_name: str, agents) -> pd.DataFrame:
    """Every line-up on this map that included all of `agents`, newest first."""
    wanted = set(agents)
    on_map = records[records["map"] == map_name]
    return on_map[on_map["agents"].map(lambda lineup: wanted <= set(lineup))]
