"""The pro-pick recommender: pick a map, lock in agents, see what the pros would pick next.

Uses the headline method from notebook 13 -- gradient boosting, half-life 45 days --
trained on every line-up in 2024 and 2025, as a recommendation for a match played
after the data ends. Each suggestion is shown with the past pro games behind it.

It describes what professional teams pick. It makes no claim that those picks win:
this project found no sign that agent picks predict the winner.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_map_table  # noqa: E402
from recommend import Classifier, games_with, lineup_records, lineup_table  # noqa: E402
from icons import glass_table, lineup_html, map_background, pick_cards  # noqa: E402

# The headline method and its settings, fixed in notebook 13 and tested in notebook 14.
HALF_LIFE_DAYS = 45

# From notebook 14, on the 2025 test half's 1,032 line-ups.
TEST_TOP3 = 0.913
TEST_TOP1 = 0.748
NAIVE_TOP3 = 0.773

# From notebook 15, on 2026 -- a new season, never used to choose or tune anything.
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


@st.cache_resource(show_spinner="Training the recommender on every line-up (about half a minute, once)…")
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

# One-click examples for the demo, so nothing has to be typed live. A callback runs before
# the widgets are drawn, which is when their values may be set.
EXAMPLES = {
    "Lotus, nothing locked": ("Lotus", []),
    "Icebox with Viper, Sova, Killjoy, Jett": ("Icebox", ["viper", "sova", "killjoy", "jett"]),
}


def load_example(map_name: str, agents: list[str]) -> None:
    st.session_state["map"] = map_name
    st.session_state["locked"] = agents


if "map" not in st.session_state:
    st.session_state["map"] = default_map

# ---------------------------------------------------------------- the controls
st.title("Meta Data")
st.caption(
    f"**What would the pros pick?** Trained on pro line-ups from February 2024 to "
    f"{data_ends:%B %Y}, recent games counting more. Tested separately on 2026."
)

with st.container(border=True, key="glass_controls"):
    with st.container(horizontal=True):
        chosen_map = st.selectbox("Map", map_names, width=220, key="map")
        locked = st.multiselect(
            "Agents already locked in",
            all_agents,
            max_selections=4,
            format_func=display,
            placeholder="Choose up to four",
            key="locked",
        )
    st.html(lineup_html(locked, slots=5 - len(locked), label=display))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption("Examples:", width="content")
        for label, (example_map, agents) in EXAMPLES.items():
            st.button(label, on_click=load_example, args=(example_map, agents),
                      icon=":material/play_arrow:", type="tertiary")
map_background(chosen_map)

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

    st.html(glass_table(
        [("Line-up", "lineup", None), ("Agents", "text", None),
         ("Times run", "text", f"Out of every pro line-up on {chosen_map} in the last "
                               f"{MOST_PLAYED_DAYS} days it was played."),
         ("Record", "text", None), ("Played most by", "text", None)],
        [[lineup, ", ".join(display(a) for a in lineup), f"{n} of {len(window)}",
          f"{int(w)}–{int(n - w)}", played_most_by(lineup)]
         for lineup, n, w in zip(common.index, common["times"], common["wins"])],
    ))
    st.caption("Real line-ups pro teams ran. Records reflect the teams too, not just the agents.")

    st.subheader("Most-played agents", divider="red")
    st.html(glass_table(
        [("", "icon", None), ("Agent", "strong", None), ("Share of line-ups", "bar", None),
         ("Line-ups", "text", None)],
        [[a, display(a), n / len(lately) * 100, f"{n} of {len(lately)}"]
         for a, n in counts.head(8).items()],
    ))
    st.info(
        "**Lock in an agent to get recommendations.** From the map alone, this most-played list "
        "is as good as anything tested.",
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
            return "No recent line-ups"
        ran_it = with_locked["agents"].map(lambda lineup: agent in lineup).sum()
        return f"{ran_it} of {len(with_locked)}"

    st.subheader("Suggested next pick", divider="red")

    def evidence(agent: str) -> str:
        share = recent_share(agent)
        if share == "No recent line-ups":
            return f"No pro line-ups on {chosen_map} in the last {EVIDENCE_WINDOW_DAYS} days had these agents"
        return f"{share} recent pro line-ups with your agents ran it"

    st.html(pick_cards([(a, share, evidence(a)) for a, share in top.head(3).items()], locked, label=display))
    if len(top) > 3:
        st.caption("Next most likely: " + ", ".join(f"{display(a)} {share:.0%}" for a, share in top.iloc[3:].items()))
    st.caption("The percentage is how strongly the recommender prefers each agent over the others still "
               "available: a share of its preference, not a chance of winning.")

    st.caption(
        f"Counts weigh all {EVIDENCE_WINDOW_DAYS} days equally; the recommender favours recent "
        f"weeks (a game {HALF_LIFE_DAYS} days old counts half), so close counts can swap order."
    )

    # greedy completion from what is locked in; with four locked it would repeat the top pick
    if len(locked) < 4:
        completion = list(locked)
        while len(completion) < 5:
            next_scores = model.scores(chosen_map, completion).drop(completion, errors="ignore")
            completion.append(next_scores.idxmax())
        st.caption("Completing the line-up one pick at a time: **"
                   + ", ".join(display(a) for a in completion) + "**")
        # Only for partial line-ups. With four locked, this "fills in the rest" measure is the
        # first suggestion alone (development data) and would contradict the tested top-3 lead
        # shown below, so it is left out.
        ours, naive = PARTIAL_LOCK[len(locked)]
        lead = (ours - naive) * 100
        st.caption(
            f"With {len(locked)} locked in, it fills in the rest "
            + (f"only just better than the most-played rule, by {lead:.1f} points" if lead < 3
               else f"{lead:.0f} points better than the most-played rule")
            + " (development data). The more you lock in, the bigger its lead."
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
            f"No pro line-up on **{chosen_map}** in the last {EVIDENCE_WINDOW_DAYS} days had all of "
            f"{locked_text}; the suggestion comes from partial matches and older games."
        )

    games = games_with(records, chosen_map, list(locked) + [explain])
    if games.empty:
        st.info(
            f"No pro team ran {display(explain)} with exactly these agents on {chosen_map} in "
            "2024–25; the suggestion comes from similar line-ups.",
            icon=":material/info:",
        )
    else:
        wins, losses = int(games["won"].sum()), int((~games["won"]).sum())
        teams = games["team"].value_counts()

        with st.container(horizontal=True):
            label = "Pro games with this line-up" if len(locked) == 4 else "Pro games with these agents"
            st.metric(f"{label}, 2024–25", f"{len(games)}", border=True,
                      help=f"Every pro game in 2024 and 2025, not only the last "
                           f"{EVIDENCE_WINDOW_DAYS} days counted in the sentence above.")
            st.metric("Their record", f"{wins}–{losses}", border=True)
            st.metric("Different teams", f"{len(teams)}", border=True)

        most = ", ".join(f"{team} ({n})" for team, n in teams.head(4).items())
        st.caption(f"Played most by: {most}.")
        st.warning(
            "**Records reflect the teams too, not just the agents.** Read them as what pros play, "
            "not what wins.",
            icon=":material/balance:",
        )

        shown = games.head(GAMES_SHOWN)
        st.html(glass_table(
            [("Date", "text", None), ("Event", "text", None), ("Team", "strong", None),
             ("Opponent", "text", None), ("Score", "text", None), ("Result", "result", None)],
            [[f"{when:%d %b %Y}".lstrip("0"), event, team, opponent, f"{f}–{a}", won]
             for when, event, team, opponent, f, a, won in zip(
                 shown["played_at"], shown["event"], shown["team"], shown["opponent"],
                 shown["rounds_for"], shown["rounds_against"], shown["won"])],
        ))
        if len(games) > GAMES_SHOWN:
            st.caption(f"The {GAMES_SHOWN} most recent of {len(games)}.")

# ---------------------------------------------------------------- how much to trust it
st.subheader("How much to trust this", divider="red")
with st.container(horizontal=True):
    st.metric("Pros' pick in its top 3", f"{TEST_TOP3:.1%}", border=True,
              help="With four agents locked in, on the second half of 2025: 1,032 line-ups it "
                   f"had never seen. It named the pick first {TEST_TOP1:.1%} of the time.")
    st.metric("On a new season, top 3", f"{NEW_SEASON_TOP3:.1%}", border=True,
              help="On 2026, a new season with new agents and an off-season in between. "
                   "Each week scored using only earlier games. 1,772 line-ups.")
    st.metric("Most-played rule, top 3", f"{NAIVE_TOP3:.1%} / {NEW_SEASON_NAIVE_TOP3:.1%}", border=True,
              help="Suggesting the most-played agents on this map lately. Scored on the rest of "
                   "2025, then on 2026.")
    st.metric("Random guess, top 3", "≈13%", border=True)

st.caption(
    "Scored week by week on games it hadn't seen, each test written down before it ran. Blind "
    "spots: brand-new agents and maps. It measures matching the pros, not winning."
)
