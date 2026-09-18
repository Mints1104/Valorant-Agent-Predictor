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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import season_folds

# What the final model is given. Combat score is only there to stand in for
# rating where rating was never recorded -- see RatingFallback.
FINAL_COLUMNS = ["rating_diff", "acs_diff", "map_picked_by_a"]


def make_model(estimator):
    """Wrap a model so gap-filling and scaling happen inside each fold."""
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        estimator,
    )


class RatingFallback(BaseEstimator, TransformerMixin):
    """Use combat score where a side's rating form is unknown.

    Rating was not recorded for all of China Kickoff and much of China Stage 1,
    so on 79 of the 756 learning maps a side's rating form is unknown, and the
    model would read that as "no difference between the teams". Combat score is
    nearly complete, so on those maps the combat-score difference is used
    instead, converted to the rating scale.

    The conversion is a single number worked out from the training rows of each
    fold, so the maps being checked never influence it. Hands on two columns:
    the repaired rating difference, and who picked the map.
    """

    def fit(self, X, y=None):
        both = X[["rating_diff", "acs_diff"]].dropna()
        # a line through zero: no gap in combat score should mean no gap in rating
        self.scale_ = float(
            (both["rating_diff"] * both["acs_diff"]).sum() / (both["acs_diff"] ** 2).sum()
        )
        return self

    def transform(self, X):
        rating = X["rating_diff"].copy()
        gap = rating.isna()
        rating.loc[gap] = X.loc[gap, "acs_diff"] * self.scale_
        return pd.DataFrame({"rating_diff": rating, "map_picked_by_a": X["map_picked_by_a"]})


def make_final_model():
    """The committed model: repaired rating difference and who picked the map, into
    logistic regression. Give it the three FINAL_COLUMNS.

    Maps where nobody on a side had played before are still unknown after the
    repair, and are filled with zero -- "no difference" -- as everywhere else.
    """
    return make_pipeline(
        RatingFallback(),
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        LogisticRegression(max_iter=5000),
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
