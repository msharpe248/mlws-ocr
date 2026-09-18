#!/usr/bin/env python3
"""Download the released model files into ``data/`` and verify them.

    .venv/bin/python scripts/fetch_models.py                 # the release matching pyproject's version
    .venv/bin/python scripts/fetch_models.py --tag v0.2.0    # a specific release
    .venv/bin/python scripts/fetch_models.py --dest /elsewhere/data

Fetches ``models-manifest.json`` and the model bundle from the GitHub
release of that tag, unpacks the bundle, and checks every file's SHA-256
against the manifest; a mismatch is an error and the file is removed.
Standard library only.  The models are the ones the profiles under
``configs/`` load; what each is, and how it was built, is in
docs/NETWORKS.md and in the manifest itself.
"""
import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "msharpe248/mlws-ocr"


def _version_from_pyproject() -> str:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.M)
    return m.group(1) if m else "0.0.0"


def _get(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default=None, help="release tag (default: v + pyproject version)")
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    tag = args.tag or f"v{_version_from_pyproject()}"
    base = f"https://github.com/{args.repo}/releases/download/{tag}/"
    args.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print(f"fetching {base}models-manifest.json")
        _get(base + "models-manifest.json", td / "manifest.json")
        manifest = json.loads((td / "manifest.json").read_text())
        bundle = td / manifest["bundle"]
        print(f"fetching {base}{manifest['bundle']}")
        _get(base + manifest["bundle"], bundle)
        if _sha256(bundle) != manifest["bundle_sha256"]:
            print("bundle checksum mismatch", file=sys.stderr)
            return 1
        with tarfile.open(bundle, "r:gz") as tar:
            names = set(tar.getnames())
            assert names == set(manifest["files"]), f"bundle/manifest disagree: {names ^ set(manifest['files'])}"
            tar.extractall(args.dest, filter="data")
    bad = 0
    for name, info in manifest["files"].items():
        p = args.dest / name
        ok = p.exists() and _sha256(p) == info["sha256"]
        print(f"  {'ok ' if ok else 'BAD'} {name:22s} {info['bytes']:>9,d} B  {info['what']}")
        if not ok:
            bad += 1
            p.unlink(missing_ok=True)
    if bad:
        print(f"{bad} file(s) failed verification and were removed", file=sys.stderr)
        return 1
    print(f"{len(manifest['files'])} model files verified in {args.dest}/ (release {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
