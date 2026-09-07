#!/usr/bin/env python3
"""Fetch a structure file from the RCSB PDB into data/structures/.

Every download is logged in data/MANIFEST.md with date, URL and SHA-256,
so the origin of each file stays traceable.

    uv run python scripts/fetch_pdb.py 2F4K
    uv run python scripts/fetch_pdb.py 2HBA --format cif
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# Anchor paths to the repo, not to the current working directory, so the
# script behaves identically no matter where it is called from.
REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = REPO_ROOT / "data" / "structures"
MANIFEST = REPO_ROOT / "data" / "MANIFEST.md"

MANIFEST_HEADER = (
    "# Data provenance\n\n"
    "Origin of every file under `data/`. Downloaded files are read-only;\n"
    "derived output belongs in `data/processed/`.\n\n"
    "| Date | File | Source | SHA-256 (first 16) |\n"
    "| --- | --- | --- | --- |\n"
)


def download(url: str) -> bytes:
    print(f"Fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            sys.exit(
                "RCSB returned 404. Check the ID; some entries are "
                "mmCIF-only, in which case try --format cif."
            )
        raise


def log_to_manifest(filename: str, url: str, digest: str) -> None:
    if not MANIFEST.exists():
        MANIFEST.write_text(MANIFEST_HEADER, encoding="utf-8")
    row = f"| {date.today().isoformat()} | `{filename}` | {url} | `{digest[:16]}` |\n"
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_id", help="four-character PDB ID, e.g. 2F4K")
    parser.add_argument("--format", choices=["pdb", "cif"], default="pdb")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing file")
    args = parser.parse_args()

    pdb_id = args.pdb_id.strip().upper()
    if len(pdb_id) != 4:
        sys.exit(f"'{pdb_id}' is not a four-character PDB ID.")

    STRUCTURES.mkdir(parents=True, exist_ok=True)
    target = STRUCTURES / f"{pdb_id}.{args.format}"

    if target.exists() and not args.force:
        sys.exit(
            f"{target.relative_to(REPO_ROOT)} already exists. "
            "Use --force to replace it."
        )

    url = f"https://files.rcsb.org/download/{pdb_id}.{args.format}"
    payload = download(url)
    digest = hashlib.sha256(payload).hexdigest()

    target.write_bytes(payload)
    log_to_manifest(target.name, url, digest)

    print(f"Wrote   {target.relative_to(REPO_ROOT)}  ({len(payload):,} bytes)")
    print(f"SHA-256 {digest}")
    print(f"Logged  {MANIFEST.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
