#!/usr/bin/env python
"""Download or copy the GSE186538 hippocampus processed data files."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
GEO_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/"
    "GSE186538/suppl"
)

DATASETS = {
    "human": (
        "GSE186538_Human_cell_meta.txt.gz",
        "GSE186538_Human_counts.mtx.gz",
        "GSE186538_Human_genes.txt.gz",
    ),
    "pig": (
        "GSE186538_Pig_cell_meta.txt.gz",
        "GSE186538_Pig_counts.mtx.gz",
        "GSE186538_Pig_genes.txt.gz",
    ),
    "macaM": (
        "GSE186538_Rhesus_cell_meta.txt.gz",
        "GSE186538_Rhesus_counts.mtx.gz",
        "GSE186538_Rhesus_genes.txt.gz",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or download GSE186538 hippocampus processed files."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Copy existing GSE186538 files from this directory instead.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Species to download or copy.",
    )
    parser.add_argument("--base-url", default=GEO_BASE_URL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_files(species: list[str]) -> list[str]:
    return [
        filename
        for species_name in species
        for filename in DATASETS[species_name]
    ]


def copy_files(
    filenames: list[str],
    source_dir: Path,
    data_dir: Path,
    overwrite: bool,
) -> None:
    for filename in filenames:
        source = source_dir / filename
        destination = data_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() and not overwrite:
            print(f"SKIP: {destination}")
            continue
        shutil.copy2(source, destination)
        print(f"COPIED: {source} -> {destination}")


def report_progress(
    block_count: int,
    block_size: int,
    total_size: int,
) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_count * block_size, total_size)
    percent = downloaded * 100.0 / total_size
    print(
        f"\r  {downloaded / 1024**2:,.1f} / "
        f"{total_size / 1024**2:,.1f} MiB ({percent:5.1f}%)",
        end="",
        flush=True,
    )


def download_files(
    filenames: list[str],
    base_url: str,
    data_dir: Path,
    overwrite: bool,
) -> None:
    for filename in filenames:
        destination = data_dir / filename
        if destination.exists() and not overwrite:
            print(f"SKIP: {destination}")
            continue
        url = f"{base_url.rstrip('/')}/{filename}"
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"DOWNLOADING: {url}")
        try:
            urllib.request.urlretrieve(url, temporary, reporthook=report_progress)
            print()
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"DOWNLOADED: {destination}")


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    filenames = selected_files(args.species)
    if args.source_dir:
        copy_files(filenames, args.source_dir, args.data_dir, args.overwrite)
    else:
        download_files(filenames, args.base_url, args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
