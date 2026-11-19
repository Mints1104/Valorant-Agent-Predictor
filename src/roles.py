"""Each agent's role, and how often pro line-ups go without one.

Roles are game knowledge, not something learned from the data: Riot gives every agent one of
four. They come from valorant-api.com (fetched November 2026), the source `fetch_icons.py`
already uses for the images. A new agent needs adding here by hand; `role` returns None for
one that isn't listed, and the app then says nothing about roles rather than guess.
"""
from collections import Counter

import pandas as pd

ROLES = ("Controller", "Initiator", "Sentinel", "Duelist")

ROLE = {
    "astra": "Controller", "brimstone": "Controller", "clove": "Controller", "harbor": "Controller",
    "miks": "Controller", "omen": "Controller", "viper": "Controller",
    "breach": "Initiator", "fade": "Initiator", "gekko": "Initiator", "kayo": "Initiator",
    "skye": "Initiator", "sova": "Initiator", "tejo": "Initiator",
    "chamber": "Sentinel", "cypher": "Sentinel", "deadlock": "Sentinel", "killjoy": "Sentinel",
    "sage": "Sentinel", "veto": "Sentinel", "vyse": "Sentinel",
    "hush": "Duelist", "iso": "Duelist", "jett": "Duelist", "neon": "Duelist", "phoenix": "Duelist",
    "raze": "Duelist", "reyna": "Duelist", "waylay": "Duelist", "yoru": "Duelist",
}

# A role is "nearly always run" if fewer than this share of pro line-ups go without it. Since
# 2024 that is the controller (no line-up without one), the initiator and the duelist; about a
# quarter of line-ups skip the sentinel, so a missing sentinel is not worth a warning.
RARELY_SKIPPED = 0.05


def role(agent: str) -> str | None:
    return ROLE.get(agent)


def role_counts(agents) -> Counter:
    """How many of the agents fill each role, in the order of ROLES, leaving out unknown agents."""
    counts = Counter(ROLE[a] for a in agents if a in ROLE)
    return Counter({r: counts[r] for r in ROLES if counts[r]})


def without_role(lineups: pd.DataFrame) -> pd.Series:
    """For each role, how many of these five-agent line-ups had no agent in it."""
    return pd.Series({r: int(lineups["agents"].map(lambda x: r not in role_counts(x)).sum())
                      for r in ROLES})


def missing_roles(agents, lineups: pd.DataFrame) -> list[tuple[str, int]]:
    """The roles a full line-up lacks that pros nearly always run, each with how many of the
    pro line-ups went without it. Empty if any agent's role is unknown, so as not to guess."""
    if any(a not in ROLE for a in agents):
        return []
    skipped = without_role(lineups)
    have = role_counts(agents)
    return [(r, int(skipped[r])) for r in ROLES
            if r not in have and skipped[r] < RARELY_SKIPPED * len(lineups)]
