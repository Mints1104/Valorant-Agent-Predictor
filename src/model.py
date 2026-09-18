"""Scores any model the same way, so different models can be compared fairly.

Every model is wrapped in the same three steps, run inside each fold:

  1. Fill gaps with zero. A missing player rating means nobody on that side had
     played before, and zero is the honest thing to say: no difference between
     the teams.
  2. Put every column on a common scale.
  3. The model itself.

Doing steps 1 and 2 *inside* the fold matters. The averages used for scaling
are worked out from the training part of each fold only; working them out
from the whole season would let information from the part being checked leak
into the part being learned from. A scikit-learn Pipeline does this
automatically, which is the point of using one rather than hand-writing the
scaling in every notebook -- as it had been, in four places.

Tree models do not need scaling, but it does them no harm, and one shape for
every model keeps the comparison honest.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import season_folds


def make_model(estimator):
    """Wrap a model so gap-filling and scaling happen inside each fold."""
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        estimator,
    )


def evaluate(pipeline, X: pd.DataFrame, y: pd.Series, maps: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and log loss on each of the five folds.

    `X`, `y` and `maps` must share an index. `maps` is needed only to cut the
    folds, which go by date and never split a match across the line.

    A fresh copy of the pipeline is fitted on every fold, so nothing learned on
    one fold can carry into the next.
    """
    if not (X.index.equals(y.index) and X.index.equals(maps.index)):
        msg = "X, y and maps must share an index, or the folds pick the wrong rows."
        raise ValueError(msg)

    # The agent columns hold whole numbers (-1, 0, +1) and the gap-filler will not
    # write into a whole-number column, so everything goes in as decimals.
    X = X.astype(float)

    rows = []
    for fold, (learn_rows, check_rows) in enumerate(season_folds(maps), start=1):
        model = clone(pipeline)
        model.fit(X.loc[learn_rows], y.loc[learn_rows])

        truth = y.loc[check_rows]
        chance_a = model.predict_proba(X.loc[check_rows])[:, 1]

        rows.append({
            "fold": fold,
            "maps": len(truth),
            "accuracy": float(((chance_a >= 0.5).astype(int) == truth).mean()),
            "log_loss": float(log_loss(truth, chance_a, labels=[0, 1])),
        })
    return pd.DataFrame(rows).set_index("fold")


def picker_baseline(y: pd.Series, maps: pd.DataFrame) -> pd.DataFrame:
    """The rule every model has to beat, scored on exactly the same folds.

    Guess whoever chose the map. The last map of a match is left over after both
    sides ban, so nobody chose it; there the rule falls back to guessing team A,
    so that it answers on every map the models answer on.

    It says a flat yes or no rather than a probability, so it has no log loss.
    """
    rows = []
    for fold, (_, check_rows) in enumerate(season_folds(maps), start=1):
        picked = maps.loc[check_rows, "map_picked_by"].map({"team_a": 1, "team_b": 0})
        guess = picked.fillna(1)
        rows.append({
            "fold": fold,
            "maps": len(check_rows),
            "accuracy": float((guess == y.loc[check_rows]).mean()),
            "log_loss": np.nan,
        })
    return pd.DataFrame(rows).set_index("fold")


def summarise(scores: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """One line for a comparison table: average, spread, and folds won against the baseline."""
    return {
        "accuracy": scores["accuracy"].mean(),
        "worst fold": scores["accuracy"].min(),
        "best fold": scores["accuracy"].max(),
        "log loss": scores["log_loss"].mean(),
        "beats baseline": int((scores["accuracy"] > baseline["accuracy"]).sum()),
    }
