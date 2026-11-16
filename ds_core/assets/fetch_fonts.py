"""Download Geist, Geist Mono and Instrument Serif (latin + latin-ext) as woff2 for offline embedding."""

import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fonts"
OUT.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Geist:wght@400;500;600"
    "&family=Geist+Mono:wght@400;500"
    "&family=Instrument+Serif:ital@0;1"
    "&display=swap"
)


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


css = get(CSS_URL).decode("utf-8")
blocks = re.findall(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*{[^}]+})", css)
kept = []
for subset, block in blocks:
    if subset not in ("latin", "latin-ext"):
        continue
    family = re.search(r"font-family:\s*'([^']+)'", block).group(1).replace(" ", "")
    style = re.search(r"font-style:\s*(\w+)", block).group(1)
    weight = re.search(r"font-weight:\s*(\d+)", block).group(1)
    url = re.search(r"url\((https://[^)]+\.woff2)\)", block).group(1)
    fname = f"{family}-{weight}-{style}-{subset}.woff2"
    (OUT / fname).write_bytes(get(url))
    kept.append(block.replace(url, fname))

(OUT / "fonts.css").write_text("\n".join(kept), encoding="utf-8")
print(f"Saved {len(kept)} font faces to {OUT}")
