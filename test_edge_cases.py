"""Stress test: feed the pipeline awkward inputs and check each is handled.

The Day 6 brief asks for missing values, out-of-range inputs, unseen categories and empty or
malformed records. Each check below names the case, what should happen, and asserts it.

    .venv\\Scripts\\python.exe test_edge_cases.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import build_map_table, split_season  # noqa: E402
from dates import load_match_dates  # noqa: E402
from features import build_features, feature_columns  # noqa: E402
from form import player_form  # noqa: E402
from model import FINAL_COLUMNS, evaluate, make_final_model  # noqa: E402
from recommend import (  # noqa: E402
    Classifier, CoOccurrence, Popularity, hide_results, lineup_records, lineup_table, pool_picks, rank_of,
)


def check(label, condition):
    assert condition, f"FAILED: {label}"
    print(f"  OK  {label}")


def raises(label, fn, error=ValueError, needle=""):
    try:
        fn()
    except error as e:
        check(label, needle in str(e))
        return
    raise AssertionError(f"FAILED: {label} (no {error.__name__} raised)")


maps = build_map_table()
learn, _ = split_season(maps)
lineups = lineup_table(learn)
as_of = lineups["played_at"].max().normalize() + pd.Timedelta(days=1)

print("1. The recommender")
recent = lineups.tail(400)
methods = {"naive rule": Popularity(7).fit(lineups, as_of),
           "co-occurrence": CoOccurrence(21).fit(lineups, as_of),
           "classifier": Classifier(45).fit(recent, as_of)}
for name, m in methods.items():
    s = m.scores("NotAMap", ["omen"])
    check(f"{name}: a map it has never seen still gets a full, finite ranking", s.notna().all() and len(s) > 0)
    s = m.scores("Lotus", ["omen", "not_an_agent"])
    check(f"{name}: an unknown locked agent is ignored", s.notna().all())
    s = m.scores("Lotus", ["omen", "omen"])
    check(f"{name}: the same agent locked twice doesn't break it", s.notna().all())
    s = m.scores("Lotus", [])
    check(f"{name}: nothing locked works", s.notna().all())
scores = methods["co-occurrence"].scores("Lotus", ["omen", "fade", "raze", "viper"])
check("an agent it has never seen counts as a miss, never a hit",
      rank_of(scores, ["omen", "fade", "raze", "viper"], "brand_new_agent") == np.inf)
check("a tie counts against the recommender",
      rank_of(pd.Series({"a": 1.0, "b": 1.0, "c": 0.5}), [], "a") == 2)
picks = pool_picks(pd.Series({"a": 0.5, "b": 0.3, "c": 0.2}), ["a"], ["a", "c"])
check("an agent pool keeps only those agents, minus the locked ones", list(picks.index) == ["c"])
check("preference shares within a pool still add up to 100%", abs(picks.sum() - 1) < 1e-9)
check("no pool means every agent still open",
      list(pool_picks(pd.Series({"a": 0.5, "b": 0.3, "c": 0.2}), ["a"]).index) == ["b", "c"])
check("a pool that is all locked in comes back empty, not with a crash",
      pool_picks(pd.Series({"a": 0.5, "b": 0.5}), ["a"], ["a"]).empty)
empty = lineups.iloc[:0]
check("counting methods cope with no history at all (empty ranking, no crash)",
      len(CoOccurrence(21).fit(empty, as_of).scores("Lotus", ["omen"])) == 0)
raises("the classifier refuses to train on no history, with a clear message",
       lambda: Classifier(45).fit(empty, as_of), ValueError, "no line-ups")

print("\n2. The win model")
X_agents, y = build_features(learn, feature_columns(maps))
form = player_form(learn, half_life=5)
X = pd.concat([X_agents[["map_picked_by_a"]], form], axis=1)[FINAL_COLUMNS].astype(float)
model = make_final_model().fit(X, y)
rows = pd.DataFrame({
    "rating_diff":     [np.nan, np.nan, 10.0, -10.0, 0.0, 1.0, -1.0],
    "acs_diff":        [30.0,   np.nan, 0.0,  0.0,   0.0, 0.0, 0.0],
    "map_picked_by_a": [0.0,    0.0,    0.0,  0.0,   1.0, 0.0, 0.0],
})
p = model.predict_proba(rows)[:, 1]
check("missing rating falls back to combat score (a 30-point edge favours team A)", p[0] > 0.5)
check("no form at all gives an even call on a decider", abs(p[1] - 0.5) < 0.02)
# Real rating gaps run from about -0.28 to +0.29. A gap of 10 is nonsense input: the model
# saturates at 100% (that's floating point, not a bug), but it must stay a valid probability.
check("absurd rating gaps (±10) stay within 0-100% and point the right way", 0 <= p[3] < 0.5 < p[2] <= 1)
check("extreme but possible gaps (±1.0) give a confident call short of certainty",
      0.9 < p[5] < 1 and 0 < p[6] < 0.1)
check("choosing the map favours the chooser", p[4] > 0.5)
unseen = learn.head(3).copy()
unseen["comp_a"] = [("miks", "omen", "sova", "viper", "yoru")] + list(unseen["comp_a"].iloc[1:])
raises("an agent the column list doesn't know is refused by name, not silently dropped",
       lambda: build_features(unseen, feature_columns(learn)), ValueError, "agent_miks")
raises("misaligned inputs are refused rather than scored on the wrong rows",
       lambda: evaluate(make_final_model(), X.iloc[1:], y, learn), ValueError, "share an index")

print("\n3. The data")
raises("a season with no date source says so", lambda: load_match_dates("vct_2023"), ValueError, "No date source")
check("every map has exactly two five-agent line-ups",
      (maps["comp_a"].map(len) == 5).all() and (maps["comp_b"].map(len) == 5).all())
check("no map is listed twice", not maps.duplicated(["match_id", "map"]).any())
every = pd.concat([build_map_table(s) for s in ("vct_2024", "vct_2025", "vct_2026")])
assets = Path(__file__).parent / "assets"
no_image = sorted({a for comp in list(every["comp_a"]) + list(every["comp_b"]) for a in comp
                   if not (assets / "agents" / f"{a}.png").exists()}
                  | {m for m in every["map"].unique() if not (assets / "maps" / "splash" / f"{m}.jpg").exists()})
check(f"every agent and map from 2024 to 2026 has an image in assets/ (missing: {no_image or 'none'})",
      not no_image)

season_2026 = build_map_table("vct_2026")
cut = season_2026["played_at"].min()
hidden = hide_results(lineup_records(pd.concat([maps, season_2026], ignore_index=True)), cut)
unopened = hidden[hidden["played_at"] >= cut]
check("hidden results leave no winner or score, and keep every line-up",
      unopened[["won", "rounds_for", "rounds_against"]].isna().all().all()
      and len(unopened) == 2 * len(season_2026))
check("earlier results stay visible", hidden[hidden["played_at"] < cut]["won"].notna().all())

print("\n4. The app, with awkward choices")
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(Path(__file__).parent / "streamlit_app.py"), default_timeout=900)
at.run()
at.selectbox[0].set_value("Icebox")                      # left the pool before 2026
at.multiselect(key="locked").set_value([])
at.run()
check("a map that left the pool shows its last line-ups", not at.exception)
at.multiselect(key="locked").set_value(["jett", "reyna", "neon", "raze"])   # four duelists
at.run()
check("a line-up no pro team would play still gets suggestions and no crash", not at.exception)
at.multiselect(key="mine").set_value(["sage", "omen"])
at.run()
check("suggestions come only from the agents you can play",
      not at.exception and "Best fit from your agents" in [s.value for s in at.subheader])
at.multiselect(key="mine").set_value(["jett"])           # already locked in
at.run()
check("if all your agents are locked in, it says so and shows everyone",
      not at.exception and any("already locked in" in i.value for i in at.info))
at.multiselect(key="locked").set_value([])
at.multiselect(key="mine").set_value(["miks"])           # barely played, and not on this old map
at.run()
check("nothing locked, with agents the pros didn't play here, doesn't crash", not at.exception)
at.multiselect(key="mine").set_value([])
at.button[1].click()                                     # the "Breeze with ..." example button
at.run()
check("the demo example button fills the map and all four agents",
      not at.exception and at.selectbox(key="map").value == "Breeze" and len(at.multiselect(key="locked").value) == 4)
for page in ("app_pages/meta.py", "app_pages/win_predictor.py"):
    at.switch_page(page)
    at.run()
    check(f"{page} loads", not at.exception)
at.multiselect[0].set_value([])                          # team A picks no agents at all
at.run()
check("win predictor: an empty line-up doesn't crash it", not at.exception)

print("\nAll edge cases handled.")
