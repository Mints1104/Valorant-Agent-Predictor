# Data sources

Which files the project uses, and how each reaches the thing being predicted: who won each
map. Paths are inside `vct_2025/` of the main dataset; other seasons use the same layout.

## Must-have

| Source | Why |
|---|---|
| `matches/maps_scores.csv` | One row per map with both scores. This is the target (`team_a_won`) |
| `matches/overview.csv` | Each player's agent per map, which gives the line-ups, and per-map stats for player form (earlier maps only) |
| `matches/draft_phase.csv` | Who picked each map: the baseline rule |
| `ids/tournaments_stages_matches_games_ids.csv` | Match and map IDs, the key that joins everything, including the dates |
| `matches/scores.csv` | Labels the five exhibition matches (`Stage == "Showmatch"`) |
| The two date datasets | The only source of dates. Without them there's no honest split |

## Nice-to-have

`agents/teams_picked_agents.csv` and `agents/agents_pick_rates.csv` give agent records and pick
rates, for the synergy question. `matches/team_mapping.csv` maps short team names to full
ones. It does **not** fix the NRG problem: it has no entry for "Mega Minors".

## Stretch

Everything recorded during a map: `eco_*`, `kills*`, `rounds_kills`, `win_loss_methods_*` and
`agents/maps_stats`. Using these to predict that map's winner is leakage. They're only usable
as a record of earlier matches.

## Ruled out: `players_stats/players_stats.csv`

It looks ideal for player form, but it's averaged per tournament stage, so a player's figure
already includes the match being predicted. It also repeats `overview.csv`'s double
counting. Player form is built from `overview.csv`'s per-map rows instead, using earlier
matches only.

## Riskiest source: the dates

The honest split depends on knowing when each match was played, and the dates come from
separate datasets joined on match IDs. If that join broke, it would break quietly: rows would
drop out and the numbers would still look plausible. So it was built first, in
`src/dates.py`, with row-count checks.
