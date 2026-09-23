"""The capstone demo. Run it with:

    streamlit run streamlit_app.py

Three pages. The agent recommender is the headline: pick a map, lock in agents, and see
what professional teams would pick next, with the past games behind each suggestion.
"How the meta moved" charts which agents pros played, month by month. The win
predictor is the supporting work, and shows live that agent picks do not predict who
wins.
"""

import streamlit as st

from app_pages.icons import RIOT_NOTICE

st.set_page_config(
    page_title="Meta Data",
    page_icon=":material/sports_esports:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page("app_pages/recommender.py", title="Agent recommender",
                icon=":material/recommend:", default=True),
        st.Page("app_pages/meta.py", title="How the meta moved",
                icon=":material/timeline:"),
        st.Page("app_pages/win_predictor.py", title="Win predictor",
                icon=":material/scoreboard:"),
    ],
    position="top",
)
page.run()

# Riot's fan-content policy asks for this wherever its game images are shown.
st.divider()
st.caption(RIOT_NOTICE)
