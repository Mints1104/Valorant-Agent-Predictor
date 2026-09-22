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
from recommend import Classifier, CoOccurrence, Popularity, lineup_table, rank_of  # noqa: E402


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

print("\n4. The app, with awkward choices")
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(Path(__file__).parent / "streamlit_app.py"), default_timeout=900)
at.run()
at.selectbox[0].set_value("Breeze")                      # left the pool after 2024
at.multiselect(key="locked").set_value([])
at.run()
check("a map that left the pool shows its last line-ups", not at.exception)
at.multiselect(key="locked").set_value(["jett", "reyna", "neon", "raze"])   # four duelists
at.run()
check("a line-up no pro team would play still gets suggestions and no crash", not at.exception)
for page in ("app_pages/meta.py", "app_pages/win_predictor.py"):
    at.switch_page(page)
    at.run()
    check(f"{page} loads", not at.exception)
at.multiselect[0].set_value([])                          # team A picks no agents at all
at.run()
check("win predictor: an empty line-up doesn't crash it", not at.exception)

print("\nAll edge cases handled.")
