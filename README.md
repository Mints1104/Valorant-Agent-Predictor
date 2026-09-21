# VALORANT Agent Recommendations & Win Prediction

A data science project. Uses VALORANT Champions Tour (VCT) esports data to
**recommend agents the way professional teams pick them**, and asks whether agent picks,
line-ups or players' form can **predict who wins**.

> Status: in progress. **The headline is a pro-pick recommender.** Given a map and four
> agents already locked in, it has the pros' actual fifth pick among its three suggestions
> **91.3%** of the time on line-ups it had never seen, against 77.3% for a naive "most-played
> lately" rule. That test was committed before it was run.
>
> Behind it, the win-prediction work: agent picks do not predict the winner, measured four
> ways and confirmed on unseen maps. Players' recent form gave a clear edge in development
> (61.1%), but on the held-back test half the final model scored **54.5% against a 52.9%
> baseline** — ahead, but within the margin of error. No line-up wins more than its agents
> deserve once team strength is accounted for. The 2026 season is untouched.

## Problem

Team comps in VALORANT esports shift with the patch meta, but it's not obvious which agent combinations actually correlate with winning at the pro level, or how much a draft (pick/ban) phase predicts a match outcome before a round is even played. This project looks at:

1. **Agent recommendations** — the headline. Given a map, and any agents already locked in, what would a professional team pick? A recommender is judged on how often it matches what the pros actually chose, on line-ups it has never seen.
2. **Composition synergy / meta-analysis** — which agent combinations, per map, are over/under-performing relative to their pick rate.
3. **Win prediction** — given the draft (map picks/bans, agent comps), how well can a model predict the match winner?

The headline moved from win prediction to recommendations on 2026-09-21, on the
instructor's advice that a pre-match win model well short of 80% should not be the main
outcome. The win-prediction work stays as the supporting evidence for why the recommender
describes what the pros *do* rather than claiming its picks win.

## Data

Two Kaggle datasets, not committed to this repo (see [`data/README.md`](data/README.md) for how to fetch them via `kagglehub`):

| Dataset | Coverage | Role |
|---|---|---|
| VCT 2021-2026 Data | 2021–2026, all regions | Primary source — match results, agent picks, and the map pick/ban phase |
| VCT 2025 All Events (Int'l + Regional) | 2025 only | Supplies the match **dates**, which the primary source has none of. The two share the same match reference numbers |
| VCT 2024 (same author as the 2025 source) | 2024 only | Supplies the 2024 match dates the same way — every 2024 match, 100%. Read from a local `vct_2024/` folder; see [`data/README.md`](data/README.md) |

Field-level definitions: [`columns_description.csv`](columns_description.csv).

## Scope

**The win model uses the 2025 season only.** When it was built, earlier years had no date
information, so their matches couldn't be put in the order they were played — and that
ordering is what stops a model being tested unfairly (see "How the model is tested" below).

**The recommender uses 2024 and 2025.** A dated 2024 season turned up later: 1,104 maps from
434 matches, every one dated. 2021–2023 are still undated. **2026 is sealed** as a second
test set for the win model and has not been looked at.

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
10. **Error analysis** — done. [`notebooks/10_error_analysis.ipynb`](notebooks/10_error_analysis.ipynb). The errors have no findable structure — the model is close to a uniform 61% model. The one real pattern is that it squashes every team toward 50%.
11. **Explainability** — done. [`notebooks/11_explainability.ipynb`](notebooks/11_explainability.ipynb). Feature importance, a partial-dependence plot, and SHAP values checked against the `shap` library. Recent form counts about twice as much as who chose the map.
12. **Win model, scored once on the test half** — done. [`notebooks/12_test_half.ipynb`](notebooks/12_test_half.ipynb), committed before it was run. 54.5% against the picker rule's 52.9%: ahead, but not distinguishable from it on 516 maps.
13. **Recommender, development** — done. [`notebooks/13_recommender_development.ipynb`](notebooks/13_recommender_development.ipynb), built by [`src/recommend.py`](src/recommend.py). Five methods compared; gradient boosting chosen as the headline by a rule written before the results.
14. **Recommender, tested once** — done. [`notebooks/14_recommender_test.ipynb`](notebooks/14_recommender_test.ipynb), committed before it was run. 91.3% top-3 against the naive rule's 77.3%.
15. **Recommender in the app** — next. Each recommendation shown with the past pro games behind it.
16. **A second win model using 2024 (v2), and scoring both on 2026** — after the app.
17. **Write-up** — what worked, what didn't, and the limitations.

## The headline: recommending agents the way the pros pick them

Give the recommender a map and four agents already locked in; it ranks every other agent by
how likely a professional team would be to pick it there. It is measured on the 2025 test
half — **1,032 line-ups it had never been scored on**, 5,160 questions — using only line-ups
played before each week, the way it would be used for real. The plan was committed before it
ran ([`notebooks/14_recommender_test.ipynb`](notebooks/14_recommender_test.ipynb)).

| With four agents locked in | Pro's pick named first | Pro's pick in the top 3 |
|---|---|---|
| **Gradient boosting — the headline** | **74.8%** (73.0–76.7) | **91.3%** (90.0–92.6) |
| Classifier with map × agent columns | 75.1% | 91.4% |
| Co-occurrence | 76.4% | 91.2% |
| Classifier, plain | 71.2% | 89.1% |
| Naive rule — the most-played agents on this map lately | 56.0% | 77.3% |
| Picking at random from the ~23 left | ~4% | ~13% |

*95% ranges from resampling whole matches.*

- **It beats the naive rule by 14.0 points of top-3** (range 12.6 to 15.5), by the rule set in
  advance.
- **The three best methods tie.** Gradient boosting is the headline because a rule written
  before the development results said so — best top-3 — and it was kept to.
- **The test scored above development (91.3% against 88.0%), but the naive rule rose by
  more** (71.1% to 77.3%). The test half was easier to predict — more history, no new
  agents — rather than the recommender getting lucky. The lead shrank slightly, from about
  17 points to 14.
- **Its value is in completing a line-up.** Given only the map, nothing beats the naive rule:
  every method recovers about 3.3–3.4 of the pros' five agents.
- **A brand-new map costs about 13 points** — 80.1% top-3 on Corrode, which arrived with no
  history, against 93.0% elsewhere.
- **It measures imitation, not winning.** It knows what the pros play; the win-prediction
  work below found that agent picks do not predict who wins.

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

**Win model on the held-back test half — the figure that counts.** Scored once, on 516 maps
no model had been scored on, with the plan committed before it ran
([`notebooks/12_test_half.ipynb`](notebooks/12_test_half.ipynb)):

| On the 516 test maps | Accuracy | Log loss |
|---|---|---|
| **Final model** | **54.5%** | 0.689 |
| Picker rule | 52.9% | — |
| Always team A | 52.3% | — |
| Agent-picks model | 50.8% | 0.704 |

It leads the picker rule by 1.6 points, with a 95% range of −3.2 to +6.4: **ahead, but not
distinguishable from the baseline on this many maps.** The agent-picks model came last again,
with a log loss worse than saying 50/50 — the main negative finding, confirmed on unseen maps.
The validation figures below were optimistic: the picker rule itself fell from 55.5% to 52.9%
on the test half, and the model's lead over it shrank from 5.6 points to 1.6.

**Development results**, measured on time-ordered validation folds cut between matches, never inside one:

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
- **The model's confidence tells you very little.** When it is 50–55% sure it is right about 60% of the time; when it is 70%+ sure, about 56%. What lined up with accuracy instead was how much history sat behind the prediction: 58.0% under 5 rated maps, 58.1% at 5–10, 59.5% at 10–20, 64.5% at 20+. **Both of those figures describe the model as it stood before the combat-score repair**, which is the model this notebook was testing. On the final model neither survives — the history ladder is gone and confidence is flat rather than falling. See the error analysis below.
- **Pulling thin records toward the average was dropped.** It was the standard fix for exactly that problem, and it did make the model's confidence more meaningful — but it cost accuracy on folds 1 and 5 and failed the rules at every strength tried. Early in the season *everyone* has a thin record, so pulling them all toward average throws away the only evidence there is. Worth revisiting with more than one season of data, where early-season players would arrive with a year of history behind them.
- **Adjusting ratings for the strength of the opponent** was considered and deliberately not tried. It is a new idea rather than a repair, and with a margin of error of ±4 to 6 points there would be no way to tell whether it helped.

The model is now frozen. It was then scored once on the test half — see the start of this section.

**Where the model gets it wrong** ([`notebooks/10_error_analysis.ipynb`](notebooks/10_error_analysis.ipynb)). **The short answer is that it has no pattern worth acting on.** The model is close to a uniform 61% model rather than a strong one in some situations and a weak one in others.

This section is a correction of an earlier version of itself, which reported that the model "predicts mismatches well and toss-ups badly". That claim did not survive a check and the check is described below, because it is the more useful part.

| | Maps | Right | ± |
|---|---|---|---|
| Teams from **different leagues** | 95 | 69.5% | 10.1 |
| **Decider** maps, which nobody chose | 103 | 68.9% | 9.7 |
| Both sides with **20 or more rated maps** of history | 135 | 65.9% | 8.4 |
| **Everything** | 629 | 61.2% | 3.9 |
| Teams from the **same league** | 534 | 59.7% | 4.2 |
| **Pacific Kickoff**, the first event of the year | 46 | 50.0% | 14.4 |

- **Read that table with the number of looks in mind.** The notebook cuts the same 629 maps about 50 ways — by tournament, by player history, by league, by who chose the map, by team. One or two groups standing out is what luck produces, not a property of the model. This is the same standard the project applies to the agents, where 1 of 27 landed outside the range chance explains and was correctly called a non-finding.
- **A bigger gap between the sides does not mean a better prediction.** Across all 629 maps the correlation between how far apart the model thought the two teams were and whether it got the map right is −0.00. That is notebook 09's confidence finding arriving from another direction, and it is the check that sinks the "mismatch" story.
- **The between-league result has no mechanism behind it.** It looked like the model doing well on mismatches. But those games were *closer* than league games, not more one-sided (a 4-round median margin against 5), and the gap the model saw between the sides was *smaller* (0.047 against 0.065). So it called closer games with less to go on more often — on 95 maps carrying a ±10 range, picked out of about 50 groups. The 69.5% is real as a measurement; the reading put on it was not.
- **Its mistakes are not just close games.** Maps it gets wrong are decided by the same median margin as maps it gets right (5 rounds), with much the same share of close games and one-sided ones. "It only loses the toss-ups" would be a flattering story and it isn't true.
- **It is not misjudging particular teams.** The teams it reads worst are ones whose maps it can't call, not ones it rates wrongly — it expected TALON to win 49% of their maps and they won 45%. That's variance, and no feature fixes it.
- **The one real finding: it squashes every team toward 50%.** Across all 26 teams with 20+ maps, the slope of what happened against what it expected is **1.52**, where 1.00 would be calibrated. Its opinion of a team spans 9 points while reality spans 27. Teams that won 55%+ of their maps are under-rated by about 8 points and teams that won 45% or less are over-rated by about the same — DRX expected 51% against an actual 62%, Paper Rex 55% against 64%. Noise in its expectations would push that slope *down*, so 1.52 is the conservative reading. This is measured on every team rather than read off a slice, which is why it stands where the others don't.

**What the model is keying on** ([`notebooks/11_explainability.ipynb`](notebooks/11_explainability.ipynb)). Two inputs, and nothing else goes in.

- **Recent form carries about twice the weight of who chose the map**, once both are put on a common scale (0.36 against 0.18). On real maps its average push is 1.6 times as large — less than 2x because who picked is nearly always at full strength, while most rating gaps are small and 81 maps have none.
- **Choosing the map is worth five points of win chance on an even matchup** — the same as a rating gap of 0.054. The median gap between two sides is 0.064, so on a typical map form counts for slightly more than the map choice, and on a close one the map choice counts for more.
- **Neither input is convincing alone.** Refitting without each: rating gap only 57.7% (beats the picker rule on 2 folds of 5), who picked only 54.4%. Together, 61.1% and all 5 folds. They carry different information.
- **The model starts every map at 50.3%** — no lean towards whichever side the data calls team A — and **both weights point the same way on every fold.**
- **SHAP values** computed by hand match the `shap` library exactly on all 756 maps. The model's most confidently wrong call looks identical, from the inside, to its most confidently right one: both inputs pointed the same way, and the other team won.

**A note on 61.1% against 61.2%, before anyone "fixes" one into the other.** 61.1% is the average of the five folds' scores and is the figure quoted everywhere else, because that is how `evaluate()` scores every model. 61.2% is the share of all 629 validation maps called correctly, pooling the folds, which is the only figure the slice table above can use — a slice cuts across folds. The folds are different sizes, so the two averages differ slightly. Both are right.

## The demo

```bash
.venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

Two pages, the recommender first.

**Agent recommender.** Pick a map and lock in up to four agents; it suggests the next pick
the way the pros make it. For any suggestion it shows the past pro games behind it: *"of the
110 pro line-ups on Icebox in the last 180 days with these four, 31 also ran Sage"*, then
how many pro maps that line-up has been played on, its record, which teams played it most,
and the most recent of those games. A caveat sits beside every record: results reflect the
teams as much as the agents, and agent picks do not predict who wins.

With **nothing** locked in, it shows no recommendations. Instead it lists the pros' most
common full line-ups on that map lately — real five-agent sets that teams ran, with how
often, their record and who ran them — and then the most-played agents, with their counts. Both notebooks found that from the map alone nothing beats
that list, and gradient boosting could misorder the obvious picks there (putting Omen fourth
on Lotus, where pros played it in every recent line-up). So that view uses no model, and its
ranking and its evidence are the same numbers.

**Win predictor.** Pick two teams, a map, and who chose it, and it predicts the winner from
the two features the win model settled on.

It loads **both** win models side by side deliberately. Changing the agent line-ups swings the
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
explanation for the 93% accuracy reported by one of the similar projects below.

## Similar projects

Three public projects on the same problem, reviewed before this one started. They are why
this project is so careful about leakage.

- **[Valorant Pro Match Analysis](https://github.com/DEF4LT-303/Valorant-Pro-Match-Analysis)** —
  random forest predicting the winning team from players' combat score, damage per round and
  economy rating. Reports **100% training accuracy and 93% test accuracy**. Those statistics
  describe how the match went, so the model is being shown the result it is asked to predict.
  This is the 93% the rest of this README refers to.
- **[ValoAI](https://github.com/Corosso/ValoAI)** — logistic regression on team performance,
  map win rates and match history, with an interface for entering two team names. Closest in
  spirit to this project, including the front end.
- **[valorant-match-predictor](https://github.com/kleinaitis/valorant-match-predictor)** —
  logistic regression on one-hot-encoded agents, from ranked matches rather than professional
  ones. Tests the agent question directly, which this project found carries no signal at the
  professional level.

## Repo structure

```
Capstone-Project/
├── README.md                     # this file
├── columns_description.csv       # data dictionary
├── requirements.txt              # project dependencies
├── test_data_pull.py             # smoke test: confirms kagglehub can fetch both datasets
├── test_form.py                  # proves the player-form feature only looks backwards
├── check_test_inputs.py          # reproduces notebook 12's input-only check: no bug in the test inputs
├── streamlit_app.py              # the demo: navigation over the two pages below
├── app_pages/
│   ├── recommender.py            # the headline: pick a map, lock in agents, see the evidence
│   └── win_predictor.py          # live win prediction, and the agent finding made visible
├── .streamlit/config.toml        # theme for the demo
├── data/
│   ├── README.md                 # how to fetch the raw data, and how dates are handled
│   └── VCT_2025_DATA_SUMMARY.md  # file-by-file notes on what each CSV contains
├── src/
│   ├── dates.py                  # match dates, bridged from the second dataset
│   ├── dataset.py                # builds the main table (one row per map played)
│   ├── features.py               # turns line-ups into numbers a model can read
│   ├── form.py                   # how the ten players had been playing, earlier matches only
│   ├── model.py                  # the final model, and scoring any model the same way, fold by fold
│   └── recommend.py              # the pro-pick recommender, and its walk-forward scoring
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
│   ├── 10_error_analysis.ipynb   # where the model is strong, and where it is guessing
│   ├── 11_explainability.ipynb   # what the win model keys on: importance, partial dependence, SHAP
│   ├── 12_test_half.ipynb        # the win model, scored once on the test half
│   ├── 13_recommender_development.ipynb # five recommender methods, and the headline choice
│   └── 14_recommender_test.ipynb # the recommender, tested once
├── vct_2024/                     # gitignored, local only -- 2024 match dates
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
