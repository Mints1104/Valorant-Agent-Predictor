"""The pro-pick recommender: pick a map, lock in agents, see what the pros would pick next.

Uses the headline method from notebook 13 -- gradient boosting, half-life 45 days --
trained on every line-up in 2024 and 2025, as a recommendation for a match played
after the data ends. Each suggestion is shown with the past pro games behind it.

It describes what professional teams pick. It makes no claim that those picks win:
this project found that agent picks do not predict the winner.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import build_map_table  # noqa: E402
from recommend import Classifier, games_with, lineup_records, lineup_table  # noqa: E402

# The headline method and its settings, fixed in notebook 13 and tested in notebook 14.
HALF_LIFE_DAYS = 45

# From notebook 14, on the 2025 test half's 1,032 line-ups.
TEST_TOP3 = 0.913
TEST_TOP1 = 0.748
NAIVE_TOP3 = 0.773

# From notebook 15, on 2026 -- a whole season it had never seen.
NEW_SEASON_TOP3 = 0.874
NEW_SEASON_NAIVE_TOP3 = 0.721

# From notebook 16, on development data only: the share of the rest of the line-up filled in
# with agents the pros actually played, by how many are locked in -- (recommender, naive rule).
PARTIAL_LOCK = {1: (0.640, 0.626), 2: (0.660, 0.593), 3: (0.676, 0.548), 4: (0.682, 0.489)}

# "Recent" in the explanations below. Longer than the model's half-life so that a
# sentence like "12 of 17 line-ups" rests on enough games to mean something.
EVIDENCE_WINDOW_DAYS = 180

# "Lately", for the most-played view shown before anything is locked in.
MOST_PLAYED_DAYS = 90

SHOWN = 5           # suggestions listed
COMMON_SHOWN = 3    # most common full line-ups listed when nothing is locked in
GAMES_SHOWN = 10    # past games listed for the chosen suggestion

DISPLAY = {"kayo": "KAY/O"}


def display(agent: str) -> str:
    return DISPLAY.get(agent, agent.capitalize())


@st.cache_data(ttl="12h", show_spinner="Loading every pro line-up from 2024 and 2025…")
def load_lineups():
    maps = pd.concat([build_map_table("vct_2024"), build_map_table()], ignore_index=True)
    return lineup_table(maps), lineup_records(maps)


@st.cache_resource(show_spinner="Training the recommender on every line-up (about a minute, once)…")
def fit_recommender(_lineups: pd.DataFrame, as_of: pd.Timestamp):
    return Classifier(
        HALF_LIFE_DAYS, estimator=HistGradientBoostingClassifier(random_state=0)
    ).fit(_lineups, as_of)


lineups, records = load_lineups()
data_ends = lineups["played_at"].max().normalize()
as_of = data_ends + pd.Timedelta(days=1)
model = fit_recommender(lineups, as_of)

recent = lineups[lineups["played_at"] >= as_of - pd.Timedelta(days=90)]
map_names = sorted(lineups["map"].unique())
default_map = recent["map"].value_counts().idxmax()
all_agents = model.agents

# ---------------------------------------------------------------- the controls
st.title("What would the pros pick?")
st.caption(
    "What would a professional team pick next? Trained on every VCT line-up from "
    f"February 2024 to {data_ends:%B %Y}, with recent games counting for more — so it "
    f"reflects the meta as of {data_ends:%B %Y}. It was tested separately on the 2026 season."
)

with st.container(border=True):
    with st.container(horizontal=True):
        chosen_map = st.selectbox("Map", map_names, index=map_names.index(default_map), width=220)
        locked = st.multiselect(
            "Agents already locked in",
            all_agents,
            max_selections=4,
            format_func=display,
            placeholder="Choose up to four",
            key="locked",
        )

# ---------------------------------------------------------------- nothing locked: just the facts
# With nothing locked in, the recommender is at its weakest -- both notebooks found that
# from the map alone nothing beats "the most-played agents here lately", and gradient
# boosting in particular can misorder the obvious picks. So this view uses no model at
# all: the ranking and the evidence are the same numbers and cannot contradict each other.
if not locked:
    on_map = lineups[lineups["map"] == chosen_map]
    last_played = on_map["played_at"].max().normalize()
    lately = on_map[on_map["played_at"] >= last_played - pd.Timedelta(days=MOST_PLAYED_DAYS)]
    counts = pd.Series([a for lineup in lately["agents"] for a in lineup]).value_counts()

    # The pros' most common full line-ups: real five-agent sets that teams actually ran,
    # so they always fit together -- five individually popular agents might not.
    # Descriptive only; no accuracy is claimed for it.
    window = records[(records["map"] == chosen_map)
                     & (records["played_at"] >= last_played - pd.Timedelta(days=MOST_PLAYED_DAYS))]
    common = (window.groupby("agents")
              .agg(times=("won", "size"), wins=("won", "sum"), latest=("played_at", "max"))
              .sort_values(["times", "latest"], ascending=False)
              .head(COMMON_SHOWN))

    st.subheader(f"Most common line-ups on {chosen_map} lately", divider="red")
    if last_played < data_ends - pd.Timedelta(days=30):
        st.caption(f"{chosen_map} has left the map pool. Last played {last_played:%d %B %Y}; "
                   f"showing its final {MOST_PLAYED_DAYS} days.")

    def played_most_by(lineup) -> str:
        teams = window[window["agents"] == lineup]["team"].value_counts()
        return ", ".join(f"{team} ({n})" for team, n in teams.head(3).items())

    st.dataframe(
        pd.DataFrame({
            "Line-up": [", ".join(display(a) for a in lineup) for lineup in common.index],
            "Times run": [f"{n} of {len(window)}" for n in common["times"]],
            "Record": [f"{int(w)}–{int(n - w)}" for w, n in zip(common["wins"], common["times"])],
            "Played most by": [played_most_by(lineup) for lineup in common.index],
        }),
        hide_index=True,
        column_config={
            "Times run": st.column_config.TextColumn(
                help=f"Out of every pro line-up on {chosen_map} in the last {MOST_PLAYED_DAYS} days "
                     "it was played."),
        },
    )
    st.caption(
        "Real line-ups that pro teams ran, so they always fit together. Records reflect the "
        "teams that ran them as much as the agents."
    )

    st.subheader("Most-played agents", divider="red")
    st.dataframe(
        pd.DataFrame({
            "Agent": [display(a) for a in counts.index],
            "Share of line-ups": (counts / len(lately)).values,
            "Line-ups": [f"{n} of {len(lately)}" for n in counts.values],
        }).head(8),
        hide_index=True,
        column_config={"Share of line-ups": st.column_config.ProgressColumn(
            format="percent", min_value=0.0, max_value=1.0)},
    )
    st.info(
        "**Lock in an agent to get recommendations.** The recommender's strength is completing "
        f"a line-up: with four agents locked in it had the pros' pick in its top three "
        f"{TEST_TOP3:.0%} of the time. From the map alone, simply listing what's most played "
        "does as well as anything tested.",
        icon=":material/lock:",
    )

# ---------------------------------------------------------------- the suggestions
if locked:
    scores = model.scores(chosen_map, locked).drop(locked, errors="ignore")
    preference = (scores / scores.sum()).sort_values(ascending=False)
    top = preference.head(SHOWN)

    window_start = as_of - pd.Timedelta(days=EVIDENCE_WINDOW_DAYS)
    in_window = records[records["played_at"] >= window_start]
    with_locked = games_with(in_window, chosen_map, locked)

    def recent_share(agent: str) -> str:
        """'12 of 17': of the recent line-ups on this map with the locked agents, how many ran this one."""
        if with_locked.empty:
            return "—"
        ran_it = with_locked["agents"].map(lambda lineup: agent in lineup).sum()
        return f"{ran_it} of {len(with_locked)}"

    st.subheader("Suggested next pick", divider="red")

    suggestions = pd.DataFrame({
        "Agent": [display(a) for a in top.index],
        "Recommender's preference": top.values,
        "Recent pro line-ups that ran it": [recent_share(a) for a in top.index],
    })
    st.dataframe(
        suggestions,
        hide_index=True,
        column_config={
            "Recommender's preference": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0,
                help="How strongly the recommender prefers each agent over the others still "
                     "available. A share of its preference, not a chance of winning.",
            ),
            "Recent pro line-ups that ran it": st.column_config.TextColumn(
                help=f"Of the pro line-ups on {chosen_map} in the last {EVIDENCE_WINDOW_DAYS} days "
                     "of data that had every agent you've locked in, how many also ran this agent.",
            ),
        },
    )

    st.caption(
        f"The counts treat every game in the last {EVIDENCE_WINDOW_DAYS} days equally. The "
        f"recommender counts recent weeks for more (a game {HALF_LIFE_DAYS} days old counts half), "
        "so when two counts are close the recommender can rank them the other way round."
    )

    # greedy completion from what is locked in, for the full picture
    completion = list(locked)
    while len(completion) < 5:
        next_scores = model.scores(chosen_map, completion).drop(completion, errors="ignore")
        completion.append(next_scores.idxmax())
    st.caption(
        "Completing the line-up one pick at a time gives: **"
        + ", ".join(display(a) for a in completion) + "**"
    )
    ours, naive = PARTIAL_LOCK[len(locked)]
    lead = (ours - naive) * 100
    st.caption(
        f"With {len(locked)} agent{'s' if len(locked) > 1 else ''} locked in, the recommender "
        f"filled in {ours:.0%} of the rest of the line-up with agents the pros actually played, "
        f"against {naive:.0%} for simply listing the most-played agents — "
        + (f"only just ahead ({lead:+.1f} points)" if lead < 3 else f"{lead:.0f} points better")
        + ". The more you lock in, the bigger its lead. Measured on development data (notebook 16)."
        + (f" With one agent left, that counts its first suggestion only; the pros' pick was in "
           f"its top three {TEST_TOP3:.0%} of the time on the test." if len(locked) == 4 else "")
    )

    # ---------------------------------------------------------------- why: the games behind it
    st.subheader("Why this pick", divider="red")

    explain = st.segmented_control(
        "Show the evidence for",
        list(top.index[:3]),
        format_func=display,
        default=top.index[0],
        key=f"explain_{chosen_map}_{'-'.join(sorted(locked))}",
    ) or top.index[0]

    locked_text = ", ".join(display(a) for a in locked)
    if not with_locked.empty:
        ran_it = with_locked["agents"].map(lambda lineup: explain in lineup).sum()
        st.markdown(
            f"Of the **{len(with_locked)}** pro line-ups on **{chosen_map}** in the last "
            f"{EVIDENCE_WINDOW_DAYS} days of data that included {locked_text}, "
            f"**{ran_it}** also ran **{display(explain)}**."
        )
    else:
        st.markdown(
            f"No pro line-up on **{chosen_map}** in the last {EVIDENCE_WINDOW_DAYS} days of data "
            f"included all of {locked_text}. The suggestion leans on line-ups that share some of "
            "them, and on older games."
        )

    games = games_with(records, chosen_map, list(locked) + [explain])
    if games.empty:
        st.info(
            f"No pro team has played {display(explain)} with exactly these agents on {chosen_map} "
            "in 2024 or 2025. The recommender is generalising from similar line-ups.",
            icon=":material/info:",
        )
    else:
        wins, losses = int(games["won"].sum()), int((~games["won"]).sum())
        teams = games["team"].value_counts()

        with st.container(horizontal=True):
            label = "Pro maps with this line-up" if len(locked) == 4 else "Pro maps with these agents"
            st.metric(label, f"{len(games)}", border=True)
            st.metric("Their record", f"{wins}–{losses}", border=True)
            st.metric("Different teams", f"{len(teams)}", border=True)

        most = ", ".join(f"{team} ({n})" for team, n in teams.head(4).items())
        st.caption(f"Played most by: {most}.")
        st.warning(
            "**Records reflect the teams as much as the agents.** A line-up played mostly by one "
            "strong team will look strong because of the team. This project found that agent "
            "picks do not predict who wins — so read this as what the pros play, not what wins.",
            icon=":material/balance:",
        )

        shown = games.head(GAMES_SHOWN)
        st.dataframe(
            pd.DataFrame({
                "Date": shown["played_at"].dt.date,
                "Event": shown["event"],
                "Team": shown["team"],
                "Opponent": shown["opponent"],
                "Score": [f"{f}–{a}" for f, a in zip(shown["rounds_for"], shown["rounds_against"])],
                "Result": ["Won" if w else "Lost" for w in shown["won"]],
            }),
            hide_index=True,
            column_config={"Date": st.column_config.DateColumn(format="D MMM YYYY")},
        )
        if len(games) > GAMES_SHOWN:
            st.caption(f"The {GAMES_SHOWN} most recent of {len(games)}.")

# ---------------------------------------------------------------- how much to trust it
st.subheader("How much to trust this", divider="red")
with st.container(horizontal=True):
    st.metric("Pro's pick in its top 3", f"{TEST_TOP3:.1%}", border=True,
              help="With four agents locked in, on the second half of 2025: 1,032 line-ups it "
                   f"had never seen. It named the pick first {TEST_TOP1:.1%} of the time.")
    st.metric("On a new season, top 3", f"{NEW_SEASON_TOP3:.1%}", border=True,
              help="On 2026 — a whole season it had never seen, with new agents and an "
                   "off-season in between. 1,772 line-ups.")
    st.metric("Naive rule, top 3", f"{NAIVE_TOP3:.1%} / {NEW_SEASON_NAIVE_TOP3:.1%}", border=True,
              help="Suggesting the most-played agents on this map lately — on the rest of 2025, "
                   "then on 2026.")
    st.metric("Random guess, top 3", "≈13%", border=True)

st.caption(
    "Measured week by week, using only games played before each week — once on the second "
    "half of 2025 and once on 2026, with each plan committed before its test ran. Its one "
    "blind spot is brand-new agents: it can't suggest an agent pros haven't played yet. It "
    "measures how well it matches what pros pick, not whether those picks win."
)
