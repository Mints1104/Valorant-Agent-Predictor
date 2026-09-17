# Data sources, and how much we need each one

Every file the project draws on, what it gives us, why we need it, and how it connects to
the thing we are predicting — which is **who won each map**.

Sources are sorted into three tiers, plus one file we have deliberately ruled out. The tier
is about how much the project depends on it, not how interesting it is.

Everything comes from two Kaggle datasets, pulled through `kagglehub` and never committed.
See [README.md](README.md) for how to sign in. Paths are relative to the `vct_2025` folder of the main dataset.

---

## Must-have

Without any one of these there is no model at all.

| Source | What it gives us | How it reaches the target |
|---|---|---|
| `matches/maps_scores.csv` | One row per map: both teams and the score | **This is the target.** `team_a_won` is worked out from the two score columns |
| `matches/overview.csv` | Every player's agent on every map, plus how they performed | The five agents each side picked — the whole of question 1, and the input to the line-up features. Also the only honest source of per-map player form (see the ruled-out file below) |
| `matches/draft_phase.csv` | The map pick and ban phase | Who chose each map. This single column is the 53.8% baseline the model has to beat |
| `ids/tournaments_stages_matches_games_ids.csv` | Match ID and Game ID for every map | The join key holding the other files together, **and** the bridge that lets dates from the second dataset attach to matches in the first |
| `matches/scores.csv` | Match-level results and the stage each match sat in | The only file that labels exhibition matches as `Stage == "Showmatch"`. Five fake matches are removed using it |
| `piyush86kumar/valorant-vct-2025-all-events` *(second dataset)* | Match dates | Nothing else in the project supplies a date. Without dates the season cannot be put in order, and without an order the train/test split is meaningless |

## Nice-to-have

Needed for the composition question, but a working model exists without them.

| Source | What it gives us | Why we want it |
|---|---|---|
| `agents/teams_picked_agents.csv` | Wins and losses per agent, per team, per map, per stage | Gives each agent an expected win rate, so a full line-up can be compared against the sum of its parts |
| `agents/agents_pick_rates.csv` | How often each agent was picked, per map | Separates "this line-up wins a lot" from "this line-up is played a lot" |
| `matches/team_mapping.csv` | 58 teams, abbreviation to full name | A safety net for matching teams by name. Worth knowing it did **not** solve our NRG problem — see below |

## Stretch

All of this describes what happened *during* a map: kills, damage, economy, how rounds were
won. Using any of it directly to predict that same map's winner is the leakage problem, and
would produce a great-looking, worthless model.

It becomes usable only when turned into a player's or team's record **across earlier
matches**, which is the plan for the second model.

`matches/eco_stats.csv`, `matches/eco_rounds.csv`, `matches/kills.csv`,
`matches/kills_stats.csv`, `matches/rounds_kills.csv`,
`matches/win_loss_methods_count.csv`, `matches/win_loss_methods_round_number.csv`,
`agents/maps_stats.csv`

`matches/maps_played.csv` is also here, but only because it duplicates the map list already
in `maps_scores.csv`.

## Ruled out — do not use

**`players_stats/players_stats.csv`.** At a glance this looks like exactly what the second
model needs: rating, average combat score, damage per round, headshot percentage, first
kills, all per player. It is a trap for two separate reasons.

1. **It is aggregated per tournament stage, not per match.** A player's rating for
   "Champions 2025 / Playoffs / Upper Quarterfinals" is their average *across that whole
   stage* — which includes the very match we would be predicting. Feeding that to a model
   is handing it the answer.
2. **It repeats the double-counting from `overview.csv`.** A player who used two agents in
   a stage gets a row for each, then a third summary row with both names in the agent
   column (`"astra, omen"`) holding the two added together.

The second model gets its player form by taking the per-map rows from `overview.csv` and
averaging each player's **earlier matches only**, calculated ourselves so the cut-off is
something we control and can test.

---

## Riskiest source, and why we dealt with it first

**The second dataset, for dates.**

Everything protecting this project from a misleadingly good result rests on knowing when
each match was played. The season is split by date so the model always learns from earlier
matches and is scored on later ones; the second model's player averages depend on knowing
which matches came before which. Take dates away and none of it holds.

And dates are the one thing the main dataset does not have — not in any of its 21 files.
So they are fetched from a completely separate dataset, uploaded by a different author, and
attached to our matches through match ID numbers. That only works because both were scraped
from the same website (vlr.gg) and therefore inherited the same reference numbers. Nothing
guarantees that stays true.

If that link ever breaks it will break *quietly*: matches lose their dates, rows drop out of
a join, and the split still produces a number that looks perfectly reasonable. That is worse
than an error.

So this was built first, in [`src/dates.py`](../src/dates.py), before any modelling. It is
also the reason the project covers only the 2025 season — the second dataset does not go
back further, so earlier years cannot be put in order at all.

---

## What surprised us

The decision log the brief asks for. Full working in `notebooks/01_eda.ipynb`.

- **`overview.csv` counts everything several times over.** Whole-match summary rows sit
  alongside real per-map rows, and each real map is split three ways (whole map, attacking
  half, defending half). A player in a three-map match has about twelve rows, not three.
  28% of the file is summary rows. The dataset's own documentation does not mention this.
- **NRG is filed under "Mega Minors"** in the team columns of `overview.csv`, but correctly
  as "NRG" everywhere else. Matching on name without fixing it loses all 25 of their matches
  silently.
- **`team_mapping.csv` does not fix that.** It lists `NRG → NRG` and has no entry for "Mega
  Minors" at all, so the alias correction had to be written by hand.
- **Match IDs are not reliably in date order.** They mostly climb over time, but there are
  182 places where a later match has a lower ID. Sorting by ID to get a timeline gives a
  wrong timeline that looks right.
- **Five exhibition matches are mixed into the season**, using invented teams that never
  appear again.
- **Several percentage columns are text**, holding values like `"91%"`. They sort and
  average wrongly without ever complaining.
- **`players_stats.csv` is stage-aggregated**, as described above. This was the biggest
  surprise of the day, because it was the file we had expected to build the second model on.
