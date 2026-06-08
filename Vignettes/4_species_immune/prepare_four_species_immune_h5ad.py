#!/usr/bin/env python
"""Convert four immune 10x archives to annotated, QC-filtered h5ad files."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import io, sparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(r"D:\111icde_addition_experiments\4_species_immune")
DEFAULT_ANNOTATION = SCRIPT_DIR / "data" / "four_species_immune_barcode_cell_type.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "processed_h5ad"

DATASETS = {
    "human": {
        "archive": "GSM5639498_Human.tar.gz",
        "archive_dir": "Human",
        "annotation_prefix": "human",
        "output": "human.h5ad",
    },
    "mouse": {
        "archive": "GSM5639494_Mouse.tar.gz",
        "archive_dir": "Mouse",
        "annotation_prefix": "mouse",
        "output": "mouse.h5ad",
    },
    "pig": {
        "archive": "GSM5639496_Pig.tar.gz",
        "archive_dir": "Pig",
        "annotation_prefix": "pig",
        "output": "pig.h5ad",
    },
    "macaM": {
        "archive": "GSM5639497_Monkey.tar.gz",
        "archive_dir": "Monkey",
        "annotation_prefix": "macaM",
        "output": "macaM.h5ad",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotated, QC-filtered h5ad files from four immune 10x archives."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    return parser.parse_args()


def strip_annotation_prefix(barcode: str) -> str:
    raw = barcode.split("_", 1)[1] if "_" in barcode else barcode
    return re.sub(r"__dup\d+$", "", raw)


def load_annotations(
    path: Path,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    exact: dict[str, str] = {}
    raw_labels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["barcode", "cell_type"]:
            raise ValueError(
                f"Expected annotation columns ['barcode', 'cell_type'], got {reader.fieldnames}"
            )
        for row in reader:
            barcode = row["barcode"]
            cell_type = row["cell_type"]
            if barcode in exact:
                raise ValueError(f"Duplicate annotation identifier: {barcode}")
            exact[barcode] = cell_type
            raw_labels[strip_annotation_prefix(barcode)].add(cell_type)
    return exact, raw_labels


def annotation_for_barcode(
    raw_barcode: str,
    prefix: str,
    exact: dict[str, str],
    raw_labels: dict[str, set[str]],
) -> tuple[str | None, str]:
    exact_key = f"{prefix}_{raw_barcode}"
    if exact_key in exact:
        return exact[exact_key], "species_prefixed_exact"
    labels = raw_labels.get(raw_barcode, set())
    if len(labels) == 1:
        return next(iter(labels)), "raw_barcode_unambiguous"
    if len(labels) > 1:
        return None, "ambiguous_cell_type"
    return None, "not_in_annotation"


def safe_extract_10x(archive_path: Path, output_dir: Path) -> Path:
    required_suffixes = {
        "barcodes.tsv.gz",
        "features.tsv.gz",
        "matrix.mtx.gz",
    }
    extracted: dict[str, Path] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            basename = Path(member.name).name
            if basename not in required_suffixes:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read {member.name} from {archive_path}")
            destination = output_dir / basename
            with destination.open("wb") as handle:
                while chunk := source.read(1024 * 1024):
                    handle.write(chunk)
            extracted[basename] = destination
    missing = required_suffixes - extracted.keys()
    if missing:
        raise ValueError(f"{archive_path} is missing 10x files: {sorted(missing)}")
    return output_dir


def read_lines_gzip(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def make_unique(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for value in values:
        count = seen.get(value, 0)
        unique.append(value if count == 0 else f"{value}-{count}")
        seen[value] = count + 1
    return unique


def read_10x_matrix(directory: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    barcodes = read_lines_gzip(directory / "barcodes.tsv.gz")
    feature_rows = [
        line.split("\t") for line in read_lines_gzip(directory / "features.tsv.gz")
    ]
    genes = make_unique(
        [row[1] if len(row) > 1 and row[1] else row[0] for row in feature_rows]
    )
    with gzip.open(directory / "matrix.mtx.gz", "rb") as handle:
        matrix = io.mmread(handle).tocsr().transpose().tocsr()
    if matrix.shape != (len(barcodes), len(genes)):
        raise ValueError(
            f"10x shape mismatch: matrix={matrix.shape}, "
            f"barcodes={len(barcodes)}, genes={len(genes)}"
        )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, barcodes, genes


def is_mito_gene(gene: str) -> bool:
    upper = gene.upper()
    return upper.startswith("MT-") or upper.startswith("MT.")


def calculate_qc(
    matrix: sparse.csr_matrix,
    genes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_counts = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mt_mask = np.asarray([is_mito_gene(gene) for gene in genes], dtype=bool)
    mt_counts = (
        np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel()
        if mt_mask.any()
        else np.zeros(matrix.shape[0], dtype=np.float32)
    )
    pct_mt = np.divide(
        mt_counts * 100.0,
        total_counts,
        out=np.zeros_like(total_counts, dtype=np.float64),
        where=total_counts > 0,
    )
    return total_counts, n_genes, pct_mt, mt_mask


def process_dataset(
    species: str,
    config: dict[str, str],
    archive_path: Path,
    output_path: Path,
    annotation_path: Path,
    exact: dict[str, str],
    raw_labels: dict[str, set[str]],
    min_genes: int,
    min_cells: int,
    max_pct_mt: float,
) -> None:
    import anndata as ad
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix=f"immune_{species}_") as temp:
        matrix_dir = safe_extract_10x(archive_path, Path(temp))
        matrix, source_barcodes, genes = read_10x_matrix(matrix_dir)

    cell_types: list[str | None] = []
    match_methods: list[str] = []
    for barcode in source_barcodes:
        cell_type, method = annotation_for_barcode(
            barcode,
            config["annotation_prefix"],
            exact,
            raw_labels,
        )
        cell_types.append(cell_type)
        match_methods.append(method)

    annotation_keep = np.asarray(
        [cell_type is not None for cell_type in cell_types], dtype=bool
    )
    matrix = matrix[annotation_keep].tocsr()
    barcodes = [
        barcode for barcode, keep in zip(source_barcodes, annotation_keep) if keep
    ]
    cell_types_kept = [
        cell_type for cell_type, keep in zip(cell_types, annotation_keep) if keep
    ]
    methods_kept = [
        method for method, keep in zip(match_methods, annotation_keep) if keep
    ]

    total_counts, n_genes, pct_mt, _ = calculate_qc(matrix, genes)
    cell_keep = (n_genes >= min_genes) & (pct_mt <= max_pct_mt)
    matrix = matrix[cell_keep].tocsr()
    barcodes = [barcode for barcode, keep in zip(barcodes, cell_keep) if keep]
    cell_types_kept = [
        cell_type for cell_type, keep in zip(cell_types_kept, cell_keep) if keep
    ]
    methods_kept = [
        method for method, keep in zip(methods_kept, cell_keep) if keep
    ]

    gene_n_cells = np.asarray((matrix > 0).sum(axis=0)).ravel()
    gene_keep = gene_n_cells >= min_cells
    matrix = matrix[:, gene_keep].tocsr()
    genes = [gene for gene, keep in zip(genes, gene_keep) if keep]

    total_counts, n_genes, pct_mt, mt_mask = calculate_qc(matrix, genes)
    unique_barcodes = [f"{config['annotation_prefix']}_{barcode}" for barcode in barcodes]
    obs = pd.DataFrame(
        {
            "cell_type": cell_types_kept,
            "species": species,
            "annotation_match": methods_kept,
            "original_barcode": barcodes,
            "total_counts": total_counts,
            "n_genes_by_counts": n_genes,
            "pct_counts_mt": pct_mt,
        },
        index=pd.Index(unique_barcodes, name="barcode"),
    )
    var = pd.DataFrame(
        {
            "n_cells_by_counts": np.asarray((matrix > 0).sum(axis=0)).ravel(),
            "mt": mt_mask,
        },
        index=pd.Index(genes, name="gene"),
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["qc_thresholds"] = {
        "min_genes": int(min_genes),
        "min_cells": int(min_cells),
        "max_pct_mt": float(max_pct_mt),
    }
    adata.uns["source_file"] = str(archive_path)
    adata.uns["annotation_file"] = str(annotation_path)
    adata.write_h5ad(output_path, compression="gzip")

    method_counts = pd.Series(methods_kept).value_counts().to_dict()
    print(
        f"{species}: source={len(source_barcodes)}, annotated={annotation_keep.sum()}, "
        f"saved={adata.n_obs} cells x {adata.n_vars} genes, "
        f"matches={method_counts} -> {output_path}"
    )


def main() -> None:
    args = parse_args()
    exact, raw_labels = load_annotations(args.annotation_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for species, config in DATASETS.items():
        archive_path = args.input_dir / config["archive"]
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        output_path = args.output_dir / config["output"]
        process_dataset(
            species=species,
            config=config,
            archive_path=archive_path,
            output_path=output_path,
            annotation_path=args.annotation_csv,
            exact=exact,
            raw_labels=raw_labels,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            max_pct_mt=args.max_pct_mt,
        )


if __name__ == "__main__":
    main()
