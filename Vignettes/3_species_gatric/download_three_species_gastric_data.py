#!/usr/bin/env python
"""Place the three processed GSE225275 expression matrices in this vignette."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
FILENAMES = {
    "human": "GSE225275_human_data.txt.gz",
    "mouse": "GSE225275_mouse_data.txt.gz",
    "pig": "GSE225275_pig_data.txt.gz",
}

# GEO does not currently expose these three prepared matrices as supplementary
# files. Add direct URLs here if the matrices are hosted in a shared repository.
DEFAULT_URLS = {
    "human": None,
    "mouse": None,
    "pig": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or download the prepared three-species gastric matrices."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory already containing the three .txt.gz matrices.",
    )
    parser.add_argument("--human-url")
    parser.add_argument("--mouse-url")
    parser.add_argument("--pig-url")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_from_source(source_dir: Path, data_dir: Path, overwrite: bool) -> None:
    for filename in FILENAMES.values():
        source = source_dir / filename
        destination = data_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() and not overwrite:
            print(f"SKIP: {destination}")
            continue
        shutil.copy2(source, destination)
        print(f"COPIED: {source} -> {destination}")


def download_files(
    data_dir: Path,
    urls: dict[str, str | None],
    overwrite: bool,
) -> None:
    missing_urls = [species for species, url in urls.items() if not url]
    if missing_urls:
        raise ValueError(
            "No direct processed-matrix URL was supplied for: "
            + ", ".join(missing_urls)
            + ". Use --source-dir or provide all three URL arguments."
        )
    for species, filename in FILENAMES.items():
        destination = data_dir / filename
        if destination.exists() and not overwrite:
            print(f"SKIP: {destination}")
            continue
        print(f"DOWNLOADING: {urls[species]}")
        urllib.request.urlretrieve(urls[species], destination)
        print(f"SAVED: {destination}")


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        copy_from_source(args.source_dir, args.data_dir, args.overwrite)
        return
    urls = {
        "human": args.human_url or DEFAULT_URLS["human"],
        "mouse": args.mouse_url or DEFAULT_URLS["mouse"],
        "pig": args.pig_url or DEFAULT_URLS["pig"],
    }
    download_files(args.data_dir, urls, args.overwrite)


if __name__ == "__main__":
    main()
