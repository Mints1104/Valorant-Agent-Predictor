# Valorant Composition & Win Prediction

A data science project. Analyzes VALORANT Champions Tour (VCT) esports data to study **agent composition synergy / meta trends** and build a **match win-prediction model**.

> Status: in progress. Both research questions now have an answer, and the model is frozen.
> A model using only agent picks scores 51.1% against a 55.5% baseline; the final model,
> built on how the ten players had been playing and who picked the map, scores 61.1%. No
> line-up wins more than its agents deserve once team strength is accounted for. A
> Streamlit app demonstrates the model live. The test half has not been touched.

## Problem

Team comps in VALORANT esports shift with the patch meta, but it's not obvious which agent combinations actually correlate with winning at the pro level, or how much a draft (pick/ban) phase predicts a match outcome before a round is even played. This project looks at:

1. **Composition synergy / meta-analysis** — which agent combinations, per map, are over/under-performing relative to their pick rate.
2. **Win prediction** — given the draft (map picks/bans, agent comps), how well can a model predict the match winner?

## Data

Two Kaggle datasets, not committed to this repo (see [`data/README.md`](data/README.md) for how to fetch them via `kagglehub`):

| Dataset | Coverage | Role |
|---|---|---|
| VCT 2021-2026 Data | 2021–2026, all regions | Primary source — match results, agent picks, and the map pick/ban phase |
| VCT 2025 All Events (Int'l + Regional) | 2025 only | Supplies the match **dates**, which the primary source has none of. The two share the same match reference numbers |

Field-level definitions: [`columns_description.csv`](columns_description.csv).

## Scope

**2025 season only.** The earlier years have no date information anywhere, so their matches
can't be put in the order they were played — and that ordering is what stops the model
being tested unfairly (see "How the model is tested" below). Expanding to earlier seasons
is a possible later step once a working model exists; it would also mean handling agents
that didn't exist yet and maps that have since left the pool.

## Approach and progress

1. **Explore and clean** — done. See [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
2. **Build the main table** — done. One row per map played: [`notebooks/02_build_map_table.ipynb`](notebooks/02_build_map_table.ipynb), assembled by [`src/dataset.py`](src/dataset.py).
3. **Turn line-ups into numbers** — done. [`notebooks/03_agent_features.ipynb`](notebooks/03_agent_features.ipynb), built by [`src/features.py`](src/features.py).
4. **First model, without player stats** — done. Logistic regression on agent picks, map and who chose the map: [`notebooks/04_baseline_model.ipynb`](notebooks/04_baseline_model.ipynb). Scores 51.4% as fitted there, which loses to the baseline. (It appears as 51.1% in the results table below — same model, fitted with the columns put on a common scale so it can be compared fairly with the player-form model. Both figures are correct.)
5. **Second model, with player form** — done. [`notebooks/06_player_form_model.ipynb`](notebooks/06_player_form_model.ipynb), built by [`src/form.py`](src/form.py) and proved leak-free by [`test_form.py`](test_form.py). Scores 59.7%, which beats the baseline.
6. **Composition synergy** — done. [`notebooks/07_composition_synergy.ipynb`](notebooks/07_composition_synergy.ipynb). No line-up over-performs once team strength is accounted for.
7. **Model families** — done. [`notebooks/08_model_families.ipynb`](notebooks/08_model_families.ipynb). Logistic regression against random forest and gradient boosting, defaults and then tuned. Logistic regression on two columns is the final model.
8. **Last feature experiments, then freeze** — done. [`notebooks/09_last_feature_experiments.ipynb`](notebooks/09_last_feature_experiments.ipynb). Combat score now stands in where rating was never recorded (kept); pulling thin records toward the average (dropped). Model frozen at 61.1%, built by `make_final_model()` in [`src/model.py`](src/model.py).
9. **Demo** — done. [`streamlit_app.py`](streamlit_app.py) predicts a map live and lets you watch the agent picks fail to matter.
10. **Error analysis** — done. [`notebooks/10_error_analysis.ipynb`](notebooks/10_error_analysis.ipynb). Where the model is strong, where it is guessing, and which weaknesses are fixable.
11. **Explainability, and the test sets** — next.
12. **Write-up** — what worked, what didn't, and the limitations.

## What we've found so far

**Problems in the raw data** (all found by checking rather than assuming, and all fixed in `src/dataset.py`):

- The player file mixes whole-match summary rows in with the real per-map rows — 28% of it. Left in, they'd count many maps twice.
- That same file lists every player three times: once for the whole map, once for each half.
- NRG is stored as "Mega Minors" in the team columns but "NRG" everywhere else. Matching teams by name without fixing this silently loses all 25 of their matches.
- Five exhibition/all-star matches use made-up teams that never play again.
- The main dataset has no dates at all. They're taken from the second dataset, which shares the same match reference numbers — see [`data/README.md`](data/README.md).

**Findings so far:**

- Team A wins 51% of maps, so anything built has to beat a coin flip to be worth having.
- The team that chose the map wins it 53.8% of the time — real, but only about four extra wins per hundred maps. Turned into a rule that also has to answer on the 18% of maps nobody picked (fall back to team A), it gets 54.6% of the whole season right. The figure models are actually judged against is 55.5%, the same rule measured on the validation folds the models are scored on.
- Which agents get played shifts heavily across the year. Tejo goes from one of the most-played agents to almost none; Omen rises from roughly one line-up in ten to one in six.
- **No single agent predicts winning.** Looking only at maps where one team had an agent and the other didn't, just 1 agent out of 27 has a win rate further from 50% than luck alone explains — and checking 27 things, that's exactly what chance produces. Teams also share about 3 of their 5 agents on a typical map, and on 99 maps the two line-ups were identical.

That last point set expectations, and it held: a model built on agent picks alone lands near a coin flip.

**Model results so far**, all measured on time-ordered validation folds cut between matches, never inside one:

| | Score |
|---|---|
| Baseline — guess whoever picked the map, team A on deciders | 55.5% |
| Agent picks + map + who picked it | **51.1%** — loses to the baseline |
| **Final model:** players' form going in (rating difference, combat score where rating is missing) + who picked it | **61.1%** — beats it on all 5 folds |

Measured by accuracy as the headline, with log loss alongside to check the probabilities are honest. Saying "50/50" to every map scores 0.693 on log loss; the final model scores 0.676, and the agent-picks model scores **0.760 — worse than claiming to know nothing**, because it commits to answers it has no grounds for.

- **The agent columns actively cost accuracy.** On their own they manage 48.0% and never clear 50.7% on any fold — they fit patterns that don't survive into the next part of the season. Added to the final model they drop it from 61.1% to 54.7%, and push its log loss to 0.729. Four separate measurements now agree.
- **A team's raw prior win rate predicts nothing** (~51%). Too noisy: 42% of matches involve a team with fewer than 10 earlier maps.
- **Players' historical ratings do predict**, and recent form matters more than old form — accuracy rises steadily as older maps are faded out.
- **No line-up wins more than its agents deserve.** Of 54 line-ups used ten or more times, two looked significant where chance alone gives 2.7, and none survived correcting for how many were tested. The apparent exceptions are teams rather than compositions: the best-looking line-up wins 65%, but Paper Rex plays 20 of its 43 maps and wins 75% with it while another team went 1–4 with the same five agents.
- **That result comes with a limit worth stating.** Detecting a realistic five-point synergy effect would need around 784 maps of one exact line-up; the most-used line-up in the season has 86. So this is not evidence synergy doesn't exist — it's evidence a realistic effect is invisible in a single season, and that anything big enough to see here is more likely a strong roster than a strong composition.

**Model families.** Random forest and gradient boosting were compared against logistic regression on the same folds, first on default settings and then tuned with twelve settings each — deliberately giving the trees more chances than logistic regression, so any bias from picking the best would favour them. This comparison was run *before* the combat-score repair below, which is why logistic regression shows 59.7% here rather than the final model's 61.1%. The repair changes one input, not the choice of model.

| | Without player stats (39) | With player stats (2) | Everything (41) |
|---|---|---|---|
| Logistic regression | 51.1% | **59.7%** | 53.9% |
| Random forest, default | 52.4% | 52.8% | 53.6% |
| Gradient boosting, default | 52.9% | 52.1% | 53.8% |
| Best of 12 tuned random forests | 53.6% | 58.2% | 58.3% |
| Best of 12 tuned gradient boosters | 56.5% | 57.6% | 57.5% |

- **On the two columns that matter, none of the 24 tuned tree settings beats untuned logistic regression.** It is the final model.
- **Default random forest is confidently wrong.** It claims 95%+ certainty on 23% of maps and gets 66 of those wrong, giving it a log loss of 1.635 — more than twice the 0.693 for knowing nothing. Logistic regression never claims more than 78%. Requiring at least 30 maps per leaf brings random forest's log loss down to 0.673, but it stays less accurate.
- **Trees beat logistic regression on the agent-heavy sets only because they are better at ignoring useless columns.** Refitting the best one without the agents costs it about one point, inside the margin of error; given the agents alone it scores 49.0%, below a coin flip. So the tree models — which *can* learn combinations of agents — find nothing in them either, confirming the synergy result by a second, independent method.

**Last feature experiments.** Two fixes, each aimed at a problem found by checking. The rules for keeping one were written down before either was run: better than the current model on at least 3 folds, worse on at most 1, and log loss no worse.

| | Accuracy | Log loss | vs the model before | |
|---|---|---|---|---|
| Model before these experiments | 59.7% | 0.678 | — | |
| **Combat score where rating is missing** | **61.1%** | **0.676** | 3 folds better, 0 worse | **kept** |
| Pulling thin records toward the average | 59.1–59.8% | 0.682–0.688 | 2 better, 2 worse | dropped |
| Both together | 59.1–59.8% | 0.676–0.677 | 3 better, 2 worse | dropped |

- **The combat-score fix is a repair, not an accuracy gain.** Rating was never recorded for all of China Kickoff and much of China Stage 1, so on 79 of the 756 learning maps the model was told "no difference between the teams" when it actually knew nothing. On the 46 of those that the folds check, the old model got 22 right — exactly as many as simply guessing whoever picked the map. With the fix it gets 26. Overall that is 9 more maps right out of 629, which is inside the margin of error; what it genuinely changes is that one region is no longer invisible to the model for part of the season.
- **The model's confidence tells you very little.** When it is 50–55% sure it is right about 60% of the time; when it is 70%+ sure, about 56%. What lined up with accuracy instead was how much history sat behind the prediction — measured **on the model as it stood before the combat-score repair**: 58.0% under 5 rated maps, 58.1% at 5–10, 59.5% at 10–20, 64.5% at 20+. After the repair that climb largely flattens, because the thin-history maps were mostly the Chinese ones it fixed. See the error analysis below for the current figures.
- **Pulling thin records toward the average was dropped.** It was the standard fix for exactly that problem, and it did make the model's confidence more meaningful — but it cost accuracy on folds 1 and 5 and failed the rules at every strength tried. Early in the season *everyone* has a thin record, so pulling them all toward average throws away the only evidence there is. Worth revisiting with more than one season of data, where early-season players would arrive with a year of history behind them.
- **Adjusting ratings for the strength of the opponent** was considered and deliberately not tried. It is a new idea rather than a repair, and with a margin of error of ±4 to 6 points there would be no way to tell whether it helped.

The model is now frozen. The test half has not been touched.

**Where the model gets it wrong** ([`notebooks/10_error_analysis.ipynb`](notebooks/10_error_analysis.ipynb)). Slicing 629 validation maps makes small groups, so each figure below carries the range chance alone could produce.

| | Maps | Right | ± |
|---|---|---|---|
| Teams from **different leagues** | 95 | 69.5% | 10.1 |
| **Decider** maps, which nobody chose | 103 | 68.9% | 9.7 |
| Both sides with **20+ rated maps** of history | 124 | 66.1% | 8.8 |
| **Everything** | 629 | 61.2% | 3.9 |
| Teams from the **same league** | 534 | 59.7% | 4.2 |
| **Pacific Kickoff**, the first event of the year | 46 | 50.0% | 14.4 |

- **It predicts mismatches well and toss-ups badly.** The four leagues only meet at Masters and Champions, and there the model gets 69.5% right against 59.7% within a league. That is not just because those matches come later, when it knows more: holding history roughly equal (both sides with 10+ rated maps) the gap widens to **72.8% against 59.5%**. Leagues differ in strength, so those matchups contain real gaps a rating difference can see. Within a league, teams are closely matched and most maps genuinely are near coin flips.
- **It does best on decider maps** — 68.9%, with no more history behind them than picked maps. A team picks the map it is best at, which partly cancels a difference in strength; the decider is neutral ground, where the stronger side shows.
- **Its mistakes are not just close games.** Maps it gets wrong are decided by the same median margin as maps it gets right (5 rounds), with much the same share of close games and one-sided ones. "It only loses the toss-ups" would be a flattering story and it isn't true.
- **It is not misjudging particular teams.** The teams it reads worst are ones whose maps it can't call, not ones it rates wrongly — it expected TALON to win 49% of their maps and they won 45%. That's variance, and no feature fixes it.
- **It does under-rate the best teams.** It expected DRX to win 51% of their maps when they won 62%, and Paper Rex 55% against 64%. Its probabilities are compressed — it rarely commits past about 70%, even for teams winning two maps in three.

## The demo

```bash
.venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

Pick two teams, a map, and who chose it, and it predicts the winner from the two features
the model settled on.

It loads **both** models side by side deliberately. Changing the agent line-ups swings the
agent-picks model by twenty points or more while the player-form model does not move at
all, because it never sees the agents. Changing a team does the reverse. That turns the
project's main finding into something you can watch happen rather than something you have
to assert — and the caption underneath points out that the number doing all the moving
belongs to the model that loses to the baseline.

The app states its own accuracy against both baselines and warns that four points over a
one-sentence rule is a lean rather than a prediction. Team ratings shown there use the
whole season, since a live prediction would be for a match played after all of it; the
model itself was trained only on the first half.

## How the model is tested

The season is split by **time**, not at random: learn from the 756 maps up to Masters
Toronto (late June), test on the 516 after it. The split falls in the gap between
tournaments, so no tournament is cut across both sides.

This matters because the game changes through the year. A random split would let the model
learn from September matches and be tested on March ones — it would already know how the
season turned out, so its score would look strong and mean nothing. This is the most likely
explanation for the 93% accuracy reported by one of the similar projects in
[`existing_projects.txt`](existing_projects.txt).

## Repo structure

```
Capstone-Project/
├── README.md                     # this file
├── project_idea_draft.txt        # original brainstorm notes
├── columns_description.csv       # data dictionary
├── requirements.txt              # project dependencies
├── test_data_pull.py             # smoke test: confirms kagglehub can fetch both datasets
├── test_form.py                  # proves the player-form feature only looks backwards
├── streamlit_app.py              # the demo: live prediction, and the agent finding made visible
├── .streamlit/config.toml        # theme for the demo
├── data/
│   ├── README.md                 # how to fetch the raw data, and how dates are handled
│   └── VCT_2025_DATA_SUMMARY.md  # file-by-file notes on what each CSV contains
├── src/
│   ├── dates.py                  # match dates, bridged from the second dataset
│   ├── dataset.py                # builds the main table (one row per map played)
│   ├── features.py               # turns line-ups into numbers a model can read
│   ├── form.py                   # how the ten players had been playing, earlier matches only
│   └── model.py                  # the final model, and scoring any model the same way, fold by fold
├── notebooks/
│   ├── 01_eda.ipynb              # exploring the raw files, and the problems in them
│   ├── 02_build_map_table.ipynb  # building and checking the main table
│   ├── 03_agent_features.ipynb   # describing line-ups as numbers
│   ├── 04_baseline_model.ipynb   # the model without player stats, and why it loses
│   ├── 05_player_form_probe.ipynb # rough check that player form predicts
│   ├── 06_player_form_model.ipynb # the model with player form, the real version
│   ├── 07_composition_synergy.ipynb # do line-ups over-perform their agents?
│   ├── 08_model_families.ipynb   # logistic regression vs tree models, and the final choice
│   ├── 09_last_feature_experiments.ipynb # combat score for missing ratings (kept), shrinkage (dropped)
│   └── 10_error_analysis.ipynb   # where the model is strong, and where it is guessing
├── Valorant_2025_All_Events_International_Regional/   # gitignored, local only
└── Valorant_Champion_Tour_2021-2026_Data/              # gitignored, local only
```

The `src/` modules are plain Python so the notebooks stay readable and the same logic can
be reused across them. Notebooks add `src/` to the import path themselves, so no install
step is needed.

Raw data folders exist locally but aren't tracked in git (see `.gitignore`) — several source CSVs exceed GitHub's 100MB file limit, so they're fetched on demand instead of versioned.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python test_data_pull.py      # confirms kagglehub can pull both datasets
```

See [`data/README.md`](data/README.md) for dataset-loading snippets.
