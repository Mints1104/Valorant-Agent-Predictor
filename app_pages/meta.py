"""How the meta moved: which agents pro teams played, map by map, month by month.

Purely descriptive -- pick rates, no results -- so it makes no claim to defend. It is
the "meta trends" half of the original brief, and the backdrop to the recommender:
the reason it counts recent games for more is visible here.
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_map_table  # noqa: E402
from recommend import lineup_table  # noqa: E402
from icons import agent_path, map_background  # noqa: E402

# Seasons shown. 2026 was added only after the recommender's 2026 test (notebook 15) ran,
# because its pick rates were that test's answers. Pick rates say nothing about who won,
# so showing them does not touch the win models' still-sealed 2026 test.
SEASONS = ["vct_2024", "vct_2025", "vct_2026"]

# A month on one map needs this many line-ups before its pick rate is shown at all.
MIN_LINEUPS = 10
MAX_AGENTS = 8

# The validated categorical palette, dark-surface steps, in fixed order.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
DISPLAY = {"kayo": "KAY/O"}
ALL_MAPS = "All maps"


def display(agent: str) -> str:
    return DISPLAY.get(agent, agent.capitalize())


@st.cache_data(ttl="12h", show_spinner="Loading every pro line-up…")
def monthly_pick_rates() -> pd.DataFrame:
    """Share of line-ups that included each agent, per map per month, plus an all-maps row."""
    lineups = pd.concat([lineup_table(build_map_table(s)) for s in SEASONS], ignore_index=True)
    lineups["month"] = lineups["played_at"].dt.to_period("M").dt.to_timestamp()
    both = pd.concat([lineups, lineups.assign(map=ALL_MAPS)], ignore_index=True)

    totals = both.groupby(["map", "month"]).size().rename("lineups")
    picks = (both.explode("agents").groupby(["map", "month", "agents"]).size()
             .rename("picked").reset_index().rename(columns={"agents": "agent"}))
    out = picks.merge(totals.reset_index(), on=["map", "month"])
    out["pick_rate"] = out["picked"] / out["lineups"]
    return out


rates = monthly_pick_rates()
first_month, last_month = rates["month"].min(), rates["month"].max()

st.title("How the meta moved")
st.caption(
    f"Share of pro line-ups that included each agent, month by month, {first_month:%B %Y} to "
    f"{last_month:%B %Y}."
)

# ---------------------------------------------------------------- biggest movers, all maps
overall = rates[rates["map"] == ALL_MAPS]
early = overall[overall["month"] < first_month + pd.DateOffset(months=6)]
late = overall[overall["month"] > last_month - pd.DateOffset(months=6)]


def average_rate(part: pd.DataFrame) -> pd.Series:
    picked = part.groupby("agent")["picked"].sum()
    return picked / part.drop_duplicates("month")["lineups"].sum()


change = pd.DataFrame({"early": average_rate(early), "late": average_rate(late)}).fillna(0.0)
change["delta"] = change["late"] - change["early"]
movers = pd.concat([change.nlargest(3, "delta"), change.nsmallest(3, "delta")])

st.subheader("Biggest movers, first six months against last six", divider="red")
with st.container(horizontal=True):
    for agent, row in movers.iterrows():
        with st.container(border=True, horizontal=True, vertical_alignment="center", gap="small",
                          width=180, key=f"glass_mover_{agent}"):
            if agent_path(agent):
                st.image(str(agent_path(agent)), width=48)
            st.metric(display(agent), f"{row['late']:.0%}", delta=f"{row['delta'] * 100:+.0f} pts",
                      help=f"{row['early']:.0%} of line-ups in the first six months, "
                           f"{row['late']:.0%} in the last six.")

# ---------------------------------------------------------------- the chart
st.subheader("Pick rate by month", divider="red")

maps_by_use = (rates[rates["map"] != ALL_MAPS].drop_duplicates(["map", "month"])
               .groupby("map")["lineups"].sum().sort_values(ascending=False).index.tolist())
with st.container(horizontal=True):
    chosen_map = st.selectbox("Map", [ALL_MAPS] + maps_by_use, width=220)
    map_background(None if chosen_map == ALL_MAPS else chosen_map)
    on_map = rates[(rates["map"] == chosen_map) & (rates["lineups"] >= MIN_LINEUPS)]
    by_use = on_map.groupby("agent")["picked"].sum().sort_values(ascending=False).index.tolist()
    # Open on the story: the biggest movers shown above, where this map has them.
    story = [a for a in movers.index if a in by_use]
    agents = st.multiselect(
        "Agents", by_use, default=story or by_use[:5], max_selections=MAX_AGENTS,
        format_func=display, key=f"meta_agents_{chosen_map}",
    )

# Colour follows the agent, not its position: an agent keeps its colour while others are
# added or removed, and a freed colour goes to the next agent added.
colour_of = st.session_state.setdefault("meta_colours", {})
for agent in list(colour_of):
    if agent not in agents:
        del colour_of[agent]
for agent in agents:
    if agent not in colour_of:
        free = [c for c in PALETTE if c not in colour_of.values()]
        colour_of[agent] = free[0]

if not agents:
    st.info("Choose at least one agent.", icon=":material/show_chart:")
else:
    shown = on_map[on_map["agent"].isin(agents)].assign(
        Agent=lambda d: d["agent"].map(display),
        Month=lambda d: d["month"],
        Share=lambda d: d["pick_rate"],
        Line_ups=lambda d: d["picked"].astype(str) + " of " + d["lineups"].astype(str),
    )
    names = [display(a) for a in agents]
    colours = [colour_of[a] for a in agents]
    chart = (
        alt.Chart(shown)
        .mark_line(point=alt.OverlayMarkDef(size=36, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("Month:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y=alt.Y("Share:Q", title="Share of pro line-ups", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Agent:N", scale=alt.Scale(domain=names, range=colours),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("Agent:N"), alt.Tooltip("Month:T", format="%B %Y"),
                     alt.Tooltip("Share:Q", format=".0%"), alt.Tooltip("Line_ups:N", title="Line-ups")],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Months with fewer than {MIN_LINEUPS} line-ups on a map are left out; lines join across "
        "the gaps between events. Hover a point for its counts."
    )

st.info(
    "**Why the recommender counts recent games for more:** a new agent can go from nothing to "
    "everywhere within weeks.",
    icon=":material/timeline:",
)
