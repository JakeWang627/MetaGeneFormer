#!/usr/bin/env python
"""Download or copy the four GSE186158 immune 10x archives."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
GEO_RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/"
    "GSE186158/suppl/GSE186158_RAW.tar"
)
ARCHIVES = {
    "GSM5639494_Mouse.tar.gz",
    "GSM5639496_Pig.tar.gz",
    "GSM5639497_Monkey.tar.gz",
    "GSM5639498_Human.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or download the four immune 10x archives from GSE186158."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing directory containing the four GSM*.tar.gz archives.",
    )
    parser.add_argument("--url", default=GEO_RAW_URL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_archives(source_dir: Path, data_dir: Path, overwrite: bool) -> None:
    for filename in sorted(ARCHIVES):
        source = source_dir / filename
        destination = data_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() and not overwrite:
            print(f"SKIP: {destination}")
            continue
        shutil.copy2(source, destination)
        print(f"COPIED: {source} -> {destination}")


def download_and_extract(url: str, data_dir: Path, overwrite: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="GSE186158_") as temp:
        raw_tar = Path(temp) / "GSE186158_RAW.tar"
        print(f"DOWNLOADING: {url}")
        urllib.request.urlretrieve(url, raw_tar)
        print(f"DOWNLOADED: {raw_tar}")

        found: set[str] = set()
        with tarfile.open(raw_tar, "r:") as archive:
            for member in archive.getmembers():
                filename = Path(member.name).name
                if filename not in ARCHIVES:
                    continue
                destination = data_dir / filename
                found.add(filename)
                if destination.exists() and not overwrite:
                    print(f"SKIP: {destination}")
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not read {member.name}")
                with destination.open("wb") as handle:
                    while chunk := source.read(1024 * 1024):
                        handle.write(chunk)
                print(f"EXTRACTED: {destination}")

        missing = ARCHIVES - found
        if missing:
            raise ValueError(f"GEO archive is missing expected files: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        copy_archives(args.source_dir, args.data_dir, args.overwrite)
    else:
        download_and_extract(args.url, args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
