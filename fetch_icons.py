"""Download agent icons and map banners from valorant-api.com into assets/ (already run).

The images belong to Riot Games and are used under its "Legal Jibber Jabber" fan-content
policy; the app shows the required notice. Run again only if a new agent or map appears.

Agent icons are 1024x1024; they are shrunk to 128x128 so the repo stays small. Map splash
images (1920x1080, the page background) become 1600-px JPEGs in assets/maps/splash/. File names match the data: lower-case agent names with
"kayo" for KAY/O, and map names as they appear in the data ("Lotus").
"""
import io
import pathlib

import requests
from PIL import Image

ROOT = pathlib.Path(__file__).parent / "assets"
AGENT_DIR, MAP_DIR, SPLASH_DIR = ROOT / "agents", ROOT / "maps", ROOT / "maps" / "splash"
for folder in (AGENT_DIR, MAP_DIR, SPLASH_DIR):
    folder.mkdir(parents=True, exist_ok=True)
OUR_MAPS = {"Abyss", "Ascent", "Bind", "Breeze", "Corrode", "Fracture", "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset"}

session = requests.Session()
agents = session.get("https://valorant-api.com/v1/agents", params={"isPlayableCharacter": "true"}, timeout=30).json()["data"]
maps = session.get("https://valorant-api.com/v1/maps", timeout=30).json()["data"]

downloaded = 0
for a in agents:
    key = a["displayName"].lower().replace("/", "")          # "KAY/O" -> "kayo"
    raw = session.get(a["displayIcon"], timeout=60).content
    downloaded += len(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGBA").resize((128, 128), Image.LANCZOS)
    img.save(AGENT_DIR / f"{key}.png", optimize=True)

found = set()
for m in maps:
    name = m["displayName"]
    if name in OUR_MAPS and name not in found and m.get("splash"):
        raw = session.get(m["splash"], timeout=120).content
        downloaded += len(raw)
        splash = Image.open(io.BytesIO(raw)).convert("RGB")
        splash.thumbnail((1600, 1600), Image.LANCZOS)
        splash.save(SPLASH_DIR / f"{name}.jpg", quality=78, optimize=True, progressive=True)
        found.add(name)

assert found == OUR_MAPS, OUR_MAPS - found
saved = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
print(f"agents {len(list(AGENT_DIR.glob('*.png')))}, maps {len(found)}; downloaded {downloaded / 1e6:.1f} MB, saved {saved / 1e6:.2f} MB")
