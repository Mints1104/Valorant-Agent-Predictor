"""A demo front end for the VCT map win-prediction model.

Placeholder for the capstone presentation. Run it with:

    streamlit run streamlit_app.py

Two models are loaded side by side on purpose. One uses how the ten players
have been performing; the other uses the agents each side picked. Changing the
agents moves the second and leaves the first alone, which is the project's main
finding made visible rather than asserted.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import build_map_table, split_season  # noqa: E402
from features import agent_list, build_features, feature_columns  # noqa: E402
from form import current_team_form, player_form  # noqa: E402
from model import FINAL_COLUMNS, make_final_model  # noqa: E402

HALF_LIFE = 5
BASELINE_ACCURACY = 0.555
FORM_ACCURACY = 0.611
AGENT_ACCURACY = 0.511

st.set_page_config(
    page_title="VCT map predictor",
    page_icon=":material/sports_esports:",
    layout="wide",
)


@st.cache_data(ttl="12h", show_spinner="Loading the 2025 season…")
def load_season():
    """The map table, each team's current form, and the model's training inputs."""
    maps = build_map_table()
    learn, _ = split_season(maps)

    columns = feature_columns(maps)
    x_agents, y = build_features(learn, columns)
    # Gaps are left in on purpose: the final model fills in missing ratings from
    # combat score itself, and filling them with zero here would hide them from it.
    form = player_form(learn, half_life=HALF_LIFE)
    teams = current_team_form(maps, half_life=HALF_LIFE)

    # each team's most-played line-up, so the agent pickers start somewhere real
    sides = pd.concat([
        maps[["team_a", "comp_a"]].rename(columns={"team_a": "team", "comp_a": "comp"}),
        maps[["team_b", "comp_b"]].rename(columns={"team_b": "team", "comp_b": "comp"}),
    ])
    favourites = (
        sides.groupby(["team", "comp"]).size().rename("uses").reset_index()
        .sort_values("uses", ascending=False).groupby("team").first()["comp"].to_dict()
    )

    return maps, learn, x_agents, y, form, teams, columns, favourites


@st.cache_resource(show_spinner="Fitting the models…")
def fit_models(_x_agents, _y, _form, columns):
    """Both models, plus the scaling each was fitted with."""
    combined = pd.concat([_x_agents, _form], axis=1)

    fitted = {
        "form": {
            "pipeline": make_final_model().fit(combined[FINAL_COLUMNS].astype(float), _y),
            "columns": FINAL_COLUMNS,
        },
    }

    x = combined[columns]
    middle, spread = x.mean(), x.std().replace(0, 1)
    model = LogisticRegression(max_iter=5000).fit((x - middle) / spread, _y)
    fitted["agents"] = {"model": model, "middle": middle, "spread": spread, "columns": columns}
    return fitted


def predict(fitted, row: dict) -> float:
    """Chance team A wins, from a single row of feature values."""
    frame = pd.DataFrame([row])[fitted["columns"]].astype(float)
    if "pipeline" in fitted:
        return float(fitted["pipeline"].predict_proba(frame)[0, 1])
    scaled = (frame - fitted["middle"]) / fitted["spread"]
    return float(fitted["model"].predict_proba(scaled)[0, 1])


maps, learn, x_agents, y, form, teams, columns, favourites = load_season()
fitted = fit_models(x_agents, y, form, columns)

team_names = list(teams.index)
map_names = sorted(maps["map"].unique())
all_agents = agent_list(maps)

# ---------------------------------------------------------------- the controls
with st.sidebar:
    st.header("The match", divider="red")

    team_a = st.selectbox("Team A", team_names, index=team_names.index("NRG")
                          if "NRG" in team_names else 0)
    team_b = st.selectbox("Team B", [t for t in team_names if t != team_a],
                          index=0)
    chosen_map = st.selectbox("Map", map_names)

    picked_by = st.segmented_control(
        "Who chose this map?",
        options=["Team A", "Team B", "Neither (decider)"],
        default="Neither (decider)",
    ) or "Neither (decider)"

    st.caption(
        "The final map of a match is whatever is left after both sides ban, "
        "so nobody picks it. That is 18% of maps."
    )

picked_value = {"Team A": 1, "Team B": -1, "Neither (decider)": 0}[picked_by]

# ---------------------------------------------------------------- the headline
st.title("Who wins this map?")
st.caption(
    "VALORANT Champions Tour 2025. The model sees only what is known before the map "
    "starts: how the ten players have been performing, and who chose the map."
)

rating_a = float(teams.loc[team_a, "rating"])
rating_b = float(teams.loc[team_b, "rating"])

chance_a = predict(fitted["form"], {
    "rating_diff": rating_a - rating_b,
    "acs_diff": float(teams.loc[team_a, "acs"]) - float(teams.loc[team_b, "acs"]),
    "map_picked_by_a": picked_value,
})

favourite, chance = (team_a, chance_a) if chance_a >= 0.5 else (team_b, 1 - chance_a)

with st.container(border=True):
    st.subheader(f"{team_a} vs {team_b} on {chosen_map}")

    with st.container(horizontal=True):
        st.metric(team_a, f"{chance_a:.1%}", border=True)
        st.metric(team_b, f"{1 - chance_a:.1%}", border=True)
        st.metric("Leaning towards", favourite,
                  delta=f"{(chance - 0.5) * 100:.1f} points above a coin flip",
                  delta_color="off", border=True)

    st.progress(chance_a, text=f"{team_a} {chance_a:.0%} — {1 - chance_a:.0%} {team_b}")

    if abs(chance_a - 0.5) < 0.03:
        st.info("Too close to call. The model is barely off a coin flip here.",
                icon=":material/balance:")

# ---------------------------------------------------------------- team detail
left, right = st.columns(2)
for column, team in [(left, team_a), (right, team_b)]:
    with column, st.container(border=True):
        st.markdown(f"**{team}**")
        row = teams.loc[team]
        with st.container(horizontal=True):
            st.metric("Player rating", f"{row['rating']:.3f}", border=True)
            st.metric("Combat score", f"{row['acs']:.0f}", border=True)
        st.caption(f"Current roster: {row['players']}")
        st.caption(f"Averaging {row['maps_played']:.0f} maps of history per player")

# ---------------------------------------------------------------- the agent demo
st.header("Now change the agents", divider="red")
st.markdown(
    "The project's main finding is that **agent picks do not predict the winner**. "
    "Pick any five agents for each side and watch what happens to the two numbers below."
)

agent_left, agent_right = st.columns(2)
with agent_left:
    comp_a = st.multiselect(f"{team_a} line-up", all_agents, max_selections=5,
                            default=list(favourites.get(team_a, all_agents[:5])),
                            key=f"comp_a_{team_a}")
with agent_right:
    comp_b = st.multiselect(f"{team_b} line-up", all_agents, max_selections=5,
                            default=list(favourites.get(team_b, all_agents[5:10])),
                            key=f"comp_b_{team_b}")

st.caption("Each side starts on its most-played line-up of the season.")

agent_row = {c: 0 for c in columns}
for agent in all_agents:
    agent_row[f"agent_{agent}"] = int(agent in comp_a) - int(agent in comp_b)
agent_row[f"map_{chosen_map}"] = 1
agent_row["map_picked_by_a"] = picked_value

agent_chance = predict(fitted["agents"], agent_row)

with st.container(horizontal=True):
    st.metric(
        "Model using player form",
        f"{chance_a:.1%}",
        delta="ignores agents entirely",
        delta_color="off",
        border=True,
        help=f"Scores {FORM_ACCURACY:.1%} — beats the baseline.",
    )
    st.metric(
        "Model using agent picks",
        f"{agent_chance:.1%}",
        delta="moves as you change agents",
        delta_color="off",
        border=True,
        help=f"Scores {AGENT_ACCURACY:.1%} — loses to the baseline.",
    )

st.caption(
    "The second number moves. That does not make it useful — it scores "
    f"{AGENT_ACCURACY:.1%}, below the {BASELINE_ACCURACY:.1%} you get by simply guessing "
    "whoever picked the map. It is reacting to patterns that do not survive into the "
    "next part of the season."
)

# ---------------------------------------------------------------- honesty
st.header("How much to trust this", divider="red")

with st.container(horizontal=True):
    st.metric("This model", f"{FORM_ACCURACY:.1%}", border=True)
    st.metric("Guess whoever picked the map", f"{BASELINE_ACCURACY:.1%}", border=True)
    st.metric("Guess at random", "50.0%", border=True)

st.warning(
    f"**About {(FORM_ACCURACY - BASELINE_ACCURACY) * 100:.0f} points better than a "
    "one-sentence rule.** These are the best 50-odd "
    "teams in the world playing a game with real randomness in it, so matches are close "
    "to coin flips and no model built on pre-match information is going to change that. "
    "Treat this as a lean, not a prediction.",
    icon=":material/warning:",
)

with st.expander("What the model actually does", icon=":material/help:"):
    st.markdown(
        f"""
Two numbers go in:

1. **The difference between the two sides' player ratings**, worked out from every map
   those ten players had played earlier in the season, with recent maps counting for more
   (a map {HALF_LIFE} maps ago counts half as much as the last one). Where a side's rating
   was never recorded — much of the Chinese league early in the season — its combat score
   is used instead, converted to the same scale.
2. **Who chose the map** — team A, team B, or nobody.

A logistic regression turns those into a probability. It was fitted on the
{len(learn):,} maps played up to late June 2025; the {len(maps) - len(learn):,} maps after
that are held back and have never been used.

**Why so few inputs?** Everything else tried made it worse. Adding the 27 agent columns
drops it from {FORM_ACCURACY:.1%} to about 55%. With only {len(learn):,} maps to learn
from, any column that does not carry its own weight actively costs accuracy.

**What it cannot see:** roster changes mid-season, who the opponent was when a rating was
earned, or anything at all about the match beyond these two numbers.
        """
    )

with st.expander("Where the numbers come from", icon=":material/database:"):
    st.markdown(
        f"""
{len(maps):,} maps from the VCT 2025 season, via two Kaggle datasets. The team ratings
shown here use the whole season, since a live prediction would be for a match played
*after* all of it. The model itself was trained only on the first half, and scored on
folds inside that half.

Ratings are missing for Chinese matches early in the season. The model uses combat score
wherever a side's rating is unknown, but the ratings in this table still rest on fewer
maps for teams from that region than their map counts suggest.
        """
    )
    st.dataframe(
        teams[["rating", "acs", "maps_played"]]
        .sort_values("rating", ascending=False)
        .rename(columns={"rating": "Player rating", "acs": "Combat score",
                         "maps_played": "Maps per player"}),
        column_config={
            "Player rating": st.column_config.NumberColumn(format="%.3f"),
            "Combat score": st.column_config.NumberColumn(format="%.0f"),
            "Maps per player": st.column_config.NumberColumn(format="%.0f"),
        },
    )
