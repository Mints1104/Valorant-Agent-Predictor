"""Checks the win model was fed normal-looking inputs on the 2025 test half.

Run after notebook 12, when the model scored lower on the test half than on
validation, to rule out a bug in how the test maps' inputs were built. It
compares what the model *sees* on the learning half and the test half:

  - the rating gap between the sides -- its centre, spread and typical size
  - how often a side has no rating form at all
  - how often nobody chose the map
  - how confident the model's calls are, and how they split between the teams

It never reads a test map's result. The model is fitted on the learning half
(whose results it needs to learn from), and on the test half only its inputs and
its own predictions are summarised -- so this can rule out broken plumbing
without becoming another look at the test score.

The printed table is the one in notebook 12's closing cell.

    python check_test_inputs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import build_map_table, split_season  # noqa: E402
from features import build_features, feature_columns  # noqa: E402
from form import player_form  # noqa: E402
from model import FINAL_COLUMNS, make_final_model  # noqa: E402


def main() -> None:
    maps = build_map_table()
    learn, test = split_season(maps)
    form = player_form(maps, half_life=5)          # whole season, backwards only, as in notebook 12
    columns = feature_columns(maps)

    x_learn, y_learn = build_features(learn, columns)
    x_test, _ = build_features(test, columns)      # the test results are discarded here, unread
    X_learn = pd.concat([x_learn, form.loc[learn.index]], axis=1)[FINAL_COLUMNS].astype(float)
    X_test = pd.concat([x_test, form.loc[test.index]], axis=1)[FINAL_COLUMNS].astype(float)

    model = make_final_model().fit(X_learn, y_learn)
    repair = model.steps[0][1]                     # the combat-score repair, as the model applies it
    gap_learn = repair.transform(X_learn)["rating_diff"]
    gap_test = repair.transform(X_test)["rating_diff"]
    chance_learn = model.predict_proba(X_learn)[:, 1]
    chance_test = model.predict_proba(X_test)[:, 1]

    rows = {
        "Rating gap, average (should be about 0)": (gap_learn.mean(), gap_test.mean(), "{:+.3f}"),
        "Rating gap, spread": (gap_learn.std(), gap_test.std(), "{:.3f}"),
        "Rating gap, median size": (gap_learn.abs().median(), gap_test.abs().median(), "{:.3f}"),
        "Maps with no rating form on a side": (gap_learn.isna().mean(), gap_test.isna().mean(), "{:.1%}"),
        "Deciders": ((X_learn["map_picked_by_a"] == 0).mean(), (X_test["map_picked_by_a"] == 0).mean(), "{:.1%}"),
        "Model's typical distance from 50%": (np.median(np.abs(chance_learn - 0.5)) * 100,
                                              np.median(np.abs(chance_test - 0.5)) * 100, "{:.1f} points"),
        "Share of maps called for team A": ((chance_learn >= 0.5).mean(), (chance_test >= 0.5).mean(), "{:.1%}"),
    }

    print(f"{'':42}{'Learning half':>15}{'Test half':>15}")
    for label, (a, b, fmt) in rows.items():
        print(f"{label:42}{fmt.format(a):>15}{fmt.format(b):>15}")
    print(f"\n{len(learn)} learning maps, {len(test)} test maps. No test result was read.")


if __name__ == "__main__":
    main()
