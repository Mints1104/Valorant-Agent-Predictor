"""The win predictor -- the project's supporting work, moved here unchanged from the old
single-page app when the agent recommender became the headline.

Two models are loaded side by side on purpose. One uses how the ten players have been
performing; the other uses the agents each side picked. Changing the agents moves the
second and leaves the first alone, which is the project's main finding made visible
rather than asserted.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_map_table, split_season  # noqa: E402
from features import agent_list, build_features, feature_columns  # noqa: E402
from form import current_team_form, player_form  # noqa: E402
from model import FINAL_COLUMNS, make_final_model  # noqa: E402
from icons import lineup_html, map_background  # noqa: E402

HALF_LIFE = 5

DISPLAY = {"kayo": "KAY/O"}


def display(agent: str) -> str:
    return DISPLAY.get(agent, agent.capitalize())


# On the held-back test half, scored once (notebook 12). These lead.
TEST_FORM_ACCURACY = 0.545
TEST_PICKER_ACCURACY = 0.529
TEST_AGENT_ACCURACY = 0.508

# On the validation folds, during development (notebooks 06-09). Quoted second.
DEV_FORM_ACCURACY = 0.611



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

    st.caption("The decider is the map left after both sides ban: 18% of games.")

picked_value = {"Team A": 1, "Team B": -1, "Neither (decider)": 0}[picked_by]
map_background(chosen_map)

# ---------------------------------------------------------------- the headline
st.title("Who wins this map?")
st.caption(
    "VCT 2025. The model sees only what's known before the map: how the ten players have been "
    "playing, and who chose the map."
)

rating_a = float(teams.loc[team_a, "rating"])
rating_b = float(teams.loc[team_b, "rating"])

chance_a = predict(fitted["form"], {
    "rating_diff": rating_a - rating_b,
    "acs_diff": float(teams.loc[team_a, "acs"]) - float(teams.loc[team_b, "acs"]),
    "map_picked_by_a": picked_value,
})

favourite, chance = (team_a, chance_a) if chance_a >= 0.5 else (team_b, 1 - chance_a)

with st.container(border=True, key="glass_headline"):
    st.subheader(f"{team_a} vs {team_b} on {chosen_map}")

    with st.container(horizontal=True):
        st.metric(team_a, f"{chance_a:.1%}", border=True)
        st.metric(team_b, f"{1 - chance_a:.1%}", border=True)
        st.metric("Leaning towards", favourite,
                  delta=f"{(chance - 0.5) * 100:.1f} points above a coin flip",
                  delta_color="off", delta_arrow="off", border=True)

    st.progress(chance_a, text=f"{team_a} {chance_a:.0%}, {team_b} {1 - chance_a:.0%}")

    if abs(chance_a - 0.5) < 0.03:
        st.info("Too close to call. The model is barely off a coin flip here.",
                icon=":material/balance:")

# ---------------------------------------------------------------- team detail
left, right = st.columns(2)
for side, (column, team) in enumerate([(left, team_a), (right, team_b)]):
    with column, st.container(border=True, key=f"glass_team_{side}"):
        st.markdown(f"**{team}**")
        row = teams.loc[team]
        with st.container(horizontal=True):
            st.metric("Player rating", f"{row['rating']:.3f}", border=True)
            st.metric("Combat score", f"{row['acs']:.0f}", border=True)
        st.caption(f"Current roster: {row['players']}")
        st.caption(f"Averaging {row['maps_played']:.0f} games of history per player")

# ---------------------------------------------------------------- the agent demo
st.header("Now change the agents", divider="red")
st.markdown("**No sign that agent picks predict the winner.** Change either line-up and watch the two "
            "numbers below.")

agent_left, agent_right = st.columns(2)
with agent_left:
    comp_a = st.multiselect(f"{team_a} line-up", all_agents, max_selections=5,
                            default=list(favourites.get(team_a, all_agents[:5])),
                            format_func=display, key=f"comp_a_{team_a}")
    st.html(lineup_html(comp_a, slots=5 - len(comp_a), size=48, label=display))
with agent_right:
    comp_b = st.multiselect(f"{team_b} line-up", all_agents, max_selections=5,
                            default=list(favourites.get(team_b, all_agents[5:10])),
                            format_func=display, key=f"comp_b_{team_b}")
    st.html(lineup_html(comp_b, slots=5 - len(comp_b), size=48, label=display))

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
        delta_arrow="off",
        border=True,
        help=f"{TEST_FORM_ACCURACY:.1%} on unseen games, against {TEST_PICKER_ACCURACY:.1%} "
             "for guessing whoever picked the map.",
    )
    st.metric(
        "Model using agent picks",
        f"{agent_chance:.1%}",
        delta="moves as you change agents",
        delta_color="off",
        delta_arrow="off",
        border=True,
        help=f"{TEST_AGENT_ACCURACY:.1%} on unseen games, below the picker rule's "
             f"{TEST_PICKER_ACCURACY:.1%}.",
    )

st.caption(
    f"The agent model moves, but on unseen games it scored {TEST_AGENT_ACCURACY:.1%}, below the "
    f"{TEST_PICKER_ACCURACY:.1%} of guessing whoever picked the map."
)

# ---------------------------------------------------------------- honesty
st.header("How much to trust this", divider="red")

with st.container(horizontal=True):
    st.metric("On unseen games", f"{TEST_FORM_ACCURACY:.1%}", border=True)
    st.metric("Guess whoever picked the map", f"{TEST_PICKER_ACCURACY:.1%}", border=True)
    st.metric("Guess at random", "50.0%", border=True)

st.caption(
    f"Scored once on 516 unseen games, with the plan written down first. In development it scored "
    f"{DEV_FORM_ACCURACY:.1%}, which was optimistic."
)

st.warning(
    f"**{(TEST_FORM_ACCURACY - TEST_PICKER_ACCURACY) * 100:.1f} points better than a one-sentence "
    "rule on unseen games: inside the margin of error.** Top teams are close to coin flips, so "
    "treat this as a lean, not a prediction.",
    icon=":material/warning:",
)

with st.expander("What the model actually does", icon=":material/help:"):
    st.markdown(
        f"""
**Two inputs:**

- **The players' rating gap**, from every game those ten players had already played that
  season, a game {HALF_LIFE} games back counting half. Combat score stands in where rating
  wasn't recorded (much of China early in the season).
- **Who chose the map:** team A, team B, or nobody.

A logistic regression turns them into a probability. Fitted on the {len(learn):,} games up to
late June 2025, then scored once on the {len(maps) - len(learn):,} after: {TEST_FORM_ACCURACY:.1%}
against {TEST_PICKER_ACCURACY:.1%} for guessing whoever picked the map.

**Why so few inputs?** Nothing else I tried helped: adding the 27 agent columns dropped it
from {DEV_FORM_ACCURACY:.1%} to about 55% in development.

**It can't see** roster changes, opponent strength, or anything else about the match.
        """
    )

with st.expander("Where the numbers come from", icon=":material/database:"):
    st.markdown(
        f"""
{len(maps):,} games from VCT 2025. The ratings below use the whole season, since a live
prediction is for a match after it; the model itself was trained on the first half only.
Chinese teams' ratings rest on fewer maps, because rating was often not recorded there.
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
