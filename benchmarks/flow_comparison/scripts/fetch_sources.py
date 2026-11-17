"""Fetch only the pinned upstream model definitions required by the preflight."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def ensure_sources(download=False):
    manifest = json.loads((ROOT / "sources.lock.json").read_text())
    for name, repo in manifest.items():
        for relative, expected in repo["files"].items():
            path = ROOT / ".sources" / name / relative
            if not path.is_file():
                if not download:
                    raise RuntimeError(f"Missing {path}; run scripts/fetch_sources.py first")
                url = f"https://raw.githubusercontent.com/{repo['repository']}/" f"{repo['commit']}/{relative}"
                request = Request(url, headers={"User-Agent": "event-flow-preflight"})
                with urlopen(request, timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise RuntimeError(f"Downloaded source checksum mismatch: {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Source changed: {path}; existing file was not overwritten")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = ensure_sources(download=not args.verify_only)
    print(json.dumps({name: repo["commit"] for name, repo in manifest.items()}, indent=2))
