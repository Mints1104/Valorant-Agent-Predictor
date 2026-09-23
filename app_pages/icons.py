"""The app's look: agent icons, the map behind each page, and "glass" tables and panels.

Images are read from assets/ so the demo works offline. They belong to Riot Games (downloaded
once by fetch_icons.py from valorant-api.com); this free, non-commercial project uses them under
Riot's "Legal Jibber Jabber" fan-content policy, which asks for RIOT_NOTICE to be shown. Agent
icons are 128 px; map splash images are 1600-px JPEGs. A missing file is not an error: callers
fall back to plain text or the plain background.
"""

import base64
import html
import io
from functools import lru_cache
from pathlib import Path

import streamlit as st
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

RIOT_NOTICE = (
    "This project was created under Riot Games' \"Legal Jibber Jabber\" policy using assets "
    "owned by Riot Games. Riot Games does not endorse or sponsor this project."
)

SLOT = "#1b2733"      # the theme's secondary background, behind each icon
RED = "#ff4655"

# "Glass": translucent, blurred panels, so tables and boxes sit on the map background instead of
# looking like solid slabs. st.dataframe paints an opaque canvas that no styling can see through,
# so tables are drawn as HTML (glass_table). Bordered containers opt in with a key starting
# "glass" (Streamlit adds the class st-key-<key>); metrics and expanders get it everywhere.
# Coloured boxes (st.info, st.warning) get a dark base under their see-through tint and off-white
# text. Captions are shown at full opacity (Streamlit dims them to 60% with `opacity`, not colour)
# with a strong shadow; dimmed grey text vanishes on a busy picture.
GLASS = "background: rgba(15, 25, 35, 0.45); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);"
GLASS_CSS = f"""<style>
[class*="st-key-glass"], [data-testid="stMetric"], [data-testid="stExpander"] details {{ {GLASS} }}
[data-testid="stAlert"] {{ background: rgba(15, 25, 35, 0.82); border-radius: 8px; }}
[data-testid="stAlert"] p {{ color: #ece8e1; }}
[class*="st-key-glass"] [data-testid="stMetric"] {{ background: none; backdrop-filter: none; -webkit-backdrop-filter: none; }}
[data-testid="stMainBlockContainer"] {{ text-shadow: 0 1px 3px rgba(0, 0, 0, 0.55); }}
[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] {{ opacity: 1; color: rgba(236, 232, 225, 0.85);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.95), 0 0 8px rgba(0, 0, 0, 0.8); }}
.glass-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 15px; color: #ece8e1;
  {GLASS} border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 12px; overflow: hidden; }}
.glass-table th {{ text-align: left; font-weight: 600; font-size: 14px; color: #a9b3bd; padding: 10px 14px;
  background: rgba(15, 25, 35, 0.55); border-bottom: 1px solid rgba(255, 255, 255, 0.10); white-space: nowrap; }}
.glass-table th .q {{ font-size: 12px; opacity: 0.7; margin-left: 4px; cursor: help; }}
.glass-table td {{ padding: 6px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); vertical-align: middle; }}
.glass-table tr:last-child td {{ border-bottom: none; }}
.glass-table td.icon img {{ width: 40px; height: 40px; border-radius: 8px; display: block; background: {SLOT}; }}
.glass-table td.lineup img {{ height: 40px; display: block; }}
.glass-table td.strong {{ font-weight: 600; }}
.glass-table td.won {{ color: #4cd07d; font-weight: 600; }}
.glass-table td.lost {{ color: {RED}; font-weight: 600; }}
.glass-table .bar {{ display: inline-block; width: 70%; max-width: 260px; height: 6px; border-radius: 3px;
  background: rgba(255, 255, 255, 0.12); vertical-align: middle; }}
.glass-table .bar div {{ height: 6px; border-radius: 3px; background: {RED}; }}
.glass-table .pct {{ margin-left: 10px; font-variant-numeric: tabular-nums; }}
</style>"""


# The selected map sits behind every page under a dark overlay: DARKNESS percent opaque at the
# top of the page, fading to DARKNESS_FADE points darker at the bottom. 55 was chosen by
# comparing 45-90 (23 Sep): the maps stay vivid, and the glass panels keep text readable.
DARKNESS = 55
DARKNESS_FADE = 16


def agent_path(agent: str) -> Path | None:
    path = ASSETS / "agents" / f"{agent}.png"
    return path if path.exists() else None


def splash_path(name: str) -> Path | None:
    path = ASSETS / "maps" / "splash" / f"{name}.jpg"
    return path if path.exists() else None


@lru_cache(maxsize=None)
def _splash_uri(name: str) -> str | None:
    path = splash_path(name)
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode() if path else None


def _background_css(name: str, level: int) -> str:
    uri = _splash_uri(name)
    if not uri:
        return ""
    top, bottom = level / 100, min(level + DARKNESS_FADE, 98) / 100
    # Darkened so text and tables stay readable; fixed so it doesn't scroll with the page.
    return GLASS_CSS + f"""<style>
[data-testid="stAppViewContainer"] {{
  background-image: linear-gradient(rgba(15, 25, 35, {top}), rgba(15, 25, 35, {bottom})), url("{uri}");
  background-size: cover; background-position: center; background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
</style>"""


def map_background(name: str | None) -> None:
    """Put the map behind the page, if it has a splash image."""
    if name and _splash_uri(name):
        st.html(_background_css(name, DARKNESS))


def _uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@lru_cache(maxsize=None)
def agent_uri(agent: str) -> str | None:
    """One agent's icon as a data URL, for image columns in st.dataframe."""
    path = agent_path(agent)
    return _uri(Image.open(path)) if path else None


@lru_cache(maxsize=None)
def lineup_uri(agents: tuple[str, ...], size: int = 64, gap: int = 6) -> str:
    """A whole line-up as one strip of icons, for a single table cell."""
    strip = Image.new("RGBA", (len(agents) * (size + gap) - gap, size), (0, 0, 0, 0))
    for i, agent in enumerate(agents):
        path = agent_path(agent)
        if path:
            icon = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
            strip.paste(icon, (i * (size + gap), 0), icon)
    return _uri(strip)


def lineup_html(agents, *, slots: int = 0, size: int = 56, label=lambda a: a) -> str:
    """Icons in a row, followed by empty slots ("?" first). Plain HTML for st.html."""
    parts = []
    box = (f"width:{size}px;height:{size}px;border-radius:8px;flex:none;"
           f"display:flex;align-items:center;justify-content:center")
    for agent in agents:
        uri = agent_uri(agent)
        if uri:
            parts.append(f'<img src="{uri}" alt="{label(agent)}" title="{label(agent)}" '
                         f'style="{box};background:{SLOT}">')
        else:
            parts.append(f'<div style="{box};background:{SLOT};font-size:12px">{label(agent)}</div>')
    for i in range(slots):
        colour = RED if i == 0 else "#44505c"
        parts.append(f'<div style="{box};border:2px dashed {colour};color:{colour};'
                     f'font-weight:700;font-size:{size // 3}px;box-sizing:border-box">'
                     f'{"?" if i == 0 else ""}</div>')
    return ('<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
            + "".join(parts) + "</div>")


def glass_table(columns, rows) -> str:
    """An HTML table for st.html that sits on the map background.

    columns: (header, kind, help) with kind one of "icon" (an agent), "lineup" (a tuple of agents),
    "bar" (a 0-100 share), "text", "strong", or "result" (True for won); help may be None.
    rows: one list of values per row, in column order.
    """
    head = "".join(
        f'<th title="{html.escape(tip)}">{html.escape(name)}<span class="q">ⓘ</span></th>' if tip
        else f"<th>{html.escape(name)}</th>" for name, _, tip in columns)
    body = []
    for row in rows:
        cells = []
        for (_, kind, _), value in zip(columns, row):
            if kind == "icon":
                uri = agent_uri(value)
                cells.append(f'<td class="icon"><img src="{uri}" alt=""></td>' if uri else "<td></td>")
            elif kind == "lineup":
                cells.append(f'<td class="lineup"><img src="{lineup_uri(tuple(value))}" alt=""></td>')
            elif kind == "bar":
                cells.append(f'<td><div class="bar"><div style="width:{max(float(value), 1):.1f}%"></div></div>'
                             f'<span class="pct">{float(value):.0f}%</span></td>')
            elif kind == "result":
                cells.append(f'<td class="{"won" if value else "lost"}">{"Won" if value else "Lost"}</td>')
            else:
                cells.append(f'<td class="{kind}">{html.escape(str(value))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (GLASS_CSS + f'<table class="glass-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')
