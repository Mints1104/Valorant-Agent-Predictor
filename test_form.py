"""Checks that player form only ever looks backwards.

Run it with the virtual environment active:

    python test_form.py

This is the test that matters most in the project. The player-form feature is
built from columns that describe what happened *during* a map -- combat scores,
damage, ratings. Those are only safe because they are turned into a record of
*earlier* matches. If that cut-off is off by even one match, the result being
predicted leaks into the prediction, the model scores brilliantly, and the whole
thing is worthless. That is the exact mistake behind the 93% accuracy reported
by one of the public projects listed in README under "Similar projects".

So rather than trusting that the code looks right, these checks prove it:

  1. Worked by hand. Pick a real map, work out both sides' averages manually
     from the matches before it, and check the function returns those numbers.
  2. The future cannot reach backwards. Build the features again from a
     truncated season and check nothing computed earlier changed. If a later
     match were leaking into an earlier one, deleting it would move the number.
  3. Nothing is known before anything is played. The very first match of the
     season has no history to draw on, so it must come back blank.
  4. Fading works. With a half-life set, recent maps count for more.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import build_map_table, split_season  # noqa: E402
from form import load_player_maps, player_form  # noqa: E402

STAT = "Average Combat Score"
SHORT = "acs"


def check_worked_by_hand(maps, players):
    """Take one real map and reproduce its numbers by hand."""
    order = ["played_at", "match_id"]

    # somewhere in the middle of the season, so both sides have a history
    target = maps.sort_values(order).iloc[len(maps) // 2]
    when = (target["played_at"], target["match_id"])

    earlier = players[
        players.apply(lambda r: (r["played_at"], r["match_id"]) < when, axis=1)
    ]
    this_match = players[players["match_id"] == target["match_id"]]

    by_hand = {}
    for side in ("team_a", "team_b"):
        team = target[side]
        names = this_match[this_match["Team"] == team]["Player"].unique()
        averages = [
            earlier[earlier["Player"] == name][STAT].dropna().mean()
            for name in names
            if len(earlier[earlier["Player"] == name][STAT].dropna())
        ]
        by_hand[side] = np.mean(averages)

    expected = by_hand["team_a"] - by_hand["team_b"]
    actual = player_form(maps).loc[target.name, f"{SHORT}_diff"]

    print(f"  map          : {target['team_a']} vs {target['team_b']}, "
          f"{target['map']}, {target['played_at']:%d %b}")
    print(f"  by hand      : {expected:+.4f}")
    print(f"  from the code: {actual:+.4f}")
    assert np.isclose(expected, actual), f"{expected} != {actual}"
    print("  OK -- the function returns the number worked out by hand")


def check_future_cannot_reach_back(maps):
    """Deleting the end of the season must not change anything before it."""
    order = ["played_at", "match_id"]
    ordered = maps.sort_values(order)
    cutoff = ordered.iloc[len(ordered) // 2]["played_at"]

    before = maps[maps["played_at"] < cutoff]

    full = player_form(maps).loc[before.index]
    truncated = player_form(before)

    columns = [c for c in full.columns if c.endswith("_diff")]
    same = np.allclose(
        full[columns].fillna(-999).to_numpy(),
        truncated[columns].fillna(-999).to_numpy(),
    )

    print(f"  cut the season at {cutoff:%d %b %Y}, leaving {len(before):,} of {len(maps):,} maps")
    print(f"  features for those maps identical either way: {same}")
    assert same, "Removing later matches changed earlier features -- the future is leaking in"
    print("  OK -- later matches have no effect on earlier features")


def check_nothing_known_at_the_start(maps):
    """The first match of the season has no history behind it."""
    form = player_form(maps)
    first = maps.sort_values(["played_at", "match_id"]).index[0]
    row = form.loc[first]

    print(f"  first map of the season: {maps.loc[first, 'played_at']:%d %b %Y}")
    print(f"  {SHORT}_diff = {row[f'{SHORT}_diff']}, games_diff = {row['games_diff']}")
    assert pd.isna(row[f"{SHORT}_diff"]), "The first map of the season claims to know something"
    assert row["games_diff"] == 0, "Nobody has played anything yet"
    print("  OK -- comes back blank, as it should")


def check_fading_changes_things(maps):
    """With a half-life set, recent maps must count for more than old ones."""
    flat = player_form(maps)[f"{SHORT}_diff"]
    faded = player_form(maps, half_life=10)[f"{SHORT}_diff"]

    both = flat.notna() & faded.notna()
    differing = (~np.isclose(flat[both], faded[both])).mean()
    correlation = flat[both].corr(faded[both])

    print(f"  maps where fading changed the number: {differing:.0%}")
    print(f"  still broadly agrees with the flat average (correlation {correlation:.2f})")
    assert differing > 0.5, "Fading barely changed anything -- is half_life being used?"
    assert correlation > 0.5, "Fading changed the numbers beyond recognition"
    print("  OK -- fading shifts the numbers without inventing new ones")


def main():
    print("Building the map table...")
    maps = build_map_table()
    learn, _ = split_season(maps)
    players = load_player_maps(learn)
    print(f"{len(learn):,} maps in the learning half, {len(players):,} player rows\n")

    checks = [
        ("1. Worked by hand", lambda: check_worked_by_hand(learn, players)),
        ("2. The future cannot reach backwards", lambda: check_future_cannot_reach_back(learn)),
        ("3. Nothing is known before anything is played",
         lambda: check_nothing_known_at_the_start(learn)),
        ("4. Fading works", lambda: check_fading_changes_things(learn)),
    ]

    for name, check in checks:
        print(name)
        check()
        print()

    print("All checks passed. The form features only look backwards.")


if __name__ == "__main__":
    main()
