# What would the pros pick?

A data science project on professional VALORANT (VCT) data. Give the recommender a map and
the agents a team has locked in, and it suggests what a pro team would pick next. The rest of
the project asks whether agent picks predict who wins. It finds no sign they do.

## The recommender

With four agents locked in, how often the pros' actual fifth pick is in its top three:

| Test | Recommender | Most-played rule: "what pros played most here lately" |
|---|---|---|
| Rest of 2025 (1,032 line-ups) | **91.3%** | 77.3% |
| 2026, a new season (1,772 line-ups) | **87.4%** | 72.1% |

- It leads the most-played rule (the notebooks call it the naive rule) by 14.0 and 15.3
  points, both with 95% ranges well above zero. It names the pick first 74.8% and 65.7% of
  the time. Random guessing gets about 13% top-3.
- The fewer agents locked in, the smaller its lead. Measured as how much of the rest of the
  line-up it fills in correctly (development data, not top 3): +1.4 points with one locked,
  +6.7 with two, +12.8 with three, +19.3 with four. With none, the most-played rule is best.
- Blind spots: a brand-new map (80% on Corrode against 93% elsewhere) and brand-new agents
  (43% top-3 on 2026's Miks and Veto, from only 77 questions).
- Gradient boosting is the headline because a rule set in advance broke a development tie
  with the map × agent classifier. Both were added after the first results, so only the tests
  count. Simple co-occurrence comes within about a point on every test, so most of the gain
  comes from using the locked agents, not the model type.
- It predicts what pros pick, not what wins.

## Does picking right win?

| On 516 unseen maps (second half of 2025) | Accuracy |
|---|---|
| Player form: the players' recent rating gap, plus who picked the map | **54.5%** |
| Picker rule: whoever chose the map wins (team A on deciders) | 52.9% |
| Agent picks, map and who picked it | 50.8% |

- **No sign that agent picks predict the winner**, tested four separate ways, and the agent
  model came last again on unseen maps. Its log loss was worse than saying 50/50 to every map
  (0.693) in development (0.760, beyond what luck explains) and no better on unseen maps
  (0.704).
- **Player form is ahead, but not provably.** In development it beat the picker rule by 5.6
  points (range +1.5 to +9.4), on all five folds (61.1% against 55.5%). On the test it led
  by 1.6 points, with a range of −3.2 to +6.4. That includes zero, and it also includes the
  development lead.
- **No sign any line-up wins more than its agents deserve.** One standout wins 65%, but Paper
  Rex, who win 66% of all their maps, play 20 of its 43. Seeing a 5-point effect would need
  about 784 maps of one line-up; the most-used has 86.
- **No pattern in its errors survives a check against luck** (notebook 10). Two first-draft
  findings were retracted: "it predicts mismatches well" and "it squashes teams toward 50%".

## How it was tested

- **By date, never at random**, so no model learns from the future. The win model learned from
  the 756 maps up to Masters Toronto (June 2025) and was tested on the 516 after it.
- **Week by week.** The recommender suggests each week using only games played before it.
- **Written down first.** Test notebooks 12, 14 and 15 were committed before they ran, then run
  once with identical cells. Nothing was changed after seeing a test result.
- **Player form only looks backwards**, which `test_form.py` proves.

## The data

| Source | Used for |
|---|---|
| [VCT 2021–2026](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) (Kaggle, version 47) | Results, line-ups, map picks. No dates |
| [VCT 2025, all events](https://www.kaggle.com/datasets/piyush86kumar/valorant-vct-2025-all-events) (version 1) | 2025 match dates |
| [VCT 2024, all events](https://www.kaggle.com/datasets/piyush86kumar/valorant-champions-tour-2024-all-events) (version 3) | 2024 match dates |
| vlr.gg, read by [`fetch_vlr_dates.py`](fetch_vlr_dates.py) | 2026 match dates (504 of 504 matched on 2025) |

2024 and 2025 (2,376 maps) are for learning, 2026 (886 maps) for testing. 2021–2023 have no
dates, so they're left out. 2026's win results are still sealed, kept for a future test of
the win model.

Problems found and fixed (in `src/dataset.py`):

- The player file counts each performance several times: whole-match summary rows (28% of
  the file) plus each half of each map.
- NRG is filed as "Mega Minors" in one file, so joining on the name silently drops all 25 of
  their matches.
- Five exhibition matches use made-up teams.
- `players_stats.csv` looks ideal for player form, but it's averaged per tournament stage, so
  it already contains the match being predicted.
- Player ratings are missing for many Chinese matches; combat score stands in.

## Limits

- It imitates the pros. Nothing here shows its picks win more.
- Win–loss records in the app reflect the teams too, and the two can't be separated.
- "No synergy" isn't proof there is none. One season is too small to see a 5-point effect.
- The app's recommender is trained on games up to October 2025.

## If it went live

| | |
|---|---|
| Scoring | Real time. Training takes about 20 seconds; after that, each recommendation is instant |
| Retraining | Weekly, as in the tests: the meta shifts and new agents arrive |
| Monitoring | Each week, score last week's pro games against the most-played rule and alert if the lead shrinks. Flag any new agent or map, its blind spots |
| Data | New dates come from vlr.gg (`fetch_vlr_dates.py`). Move the pinned dataset versions on deliberately, re-running `test_form.py` and `test_edge_cases.py` |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name valorant-capstone   # the kernel the notebooks use
python test_data_pull.py    # downloads the three datasets; needs a Kaggle token (data/README.md)
python test_form.py         # proves player form only looks backwards
python test_edge_cases.py   # feeds the pipeline awkward inputs; each must be handled
streamlit run streamlit_app.py
```

The app has three pages: the recommender, with the past pro games behind each suggestion;
how the meta moved, month by month; and the win predictor. Dataset versions are pinned, so
after the first download everything runs offline.

## The notebooks

| | |
|---|---|
| 01 | First look at the raw files, and the problems in them |
| 02 | The main table: one row per map |
| 03 | Turning line-ups into numbers |
| 04 | Agent picks model, which doesn't beat the baseline |
| 05–06 | Player form model, which beats it |
| 07 | Do line-ups over-perform? No sign |
| 08 | Logistic regression against tree models |
| 09 | The last two fixes, then the win model is frozen |
| 10 | Where the win model goes wrong |
| 11 | What the win model keys on (SHAP) |
| 12 | Win model, tested once |
| 13 | Recommender: five methods compared |
| 14 | Recommender, tested once on the rest of 2025 |
| 15 | Recommender, tested once on 2026 |
| 16 | Recommender with one to three agents locked in |

Code is in `src/`, the app in `streamlit_app.py` and `app_pages/`, the deck in
`presentation/`, with screenshots of the demo in `presentation/demo_fallback/`.
[`progress/START_HERE.md`](progress/START_HERE.md) is a one-page cheat sheet.

## Similar projects

- [Valorant Pro Match Analysis](https://github.com/DEF4LT-303/Valorant-Pro-Match-Analysis)
  reports 93% by predicting the winner from each player's kills, damage and combat score in the
  game being predicted. That is the leak this project avoids.
- [ValoAI](https://github.com/Corosso/ValoAI): logistic regression on team performance. The
  closest in spirit, including the app.
- [valorant-match-predictor](https://github.com/kleinaitis/valorant-match-predictor): agent
  picks in ranked games, not pro ones.
