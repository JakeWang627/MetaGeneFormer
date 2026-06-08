#!/usr/bin/env python
"""Convert three hippocampus matrices to annotated, QC-filtered h5ad files."""

from __future__ import annotations

import argparse
import csv
import gzip
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(r"D:\111icde_addition_experiments\3_species_hippocampus")
DEFAULT_ANNOTATION = (
    SCRIPT_DIR / "data" / "three_species_hippocampus_barcode_cell_type.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "processed_h5ad"

# Human is intentionally included. Its matrix is currently unavailable, so the
# default behavior is to skip it until GSE186538_Human_counts.mtx.gz is added.
DATASETS = {
    "human": {
        "source_name": "Human",
        "matrix": "GSE186538_Human_counts.mtx.gz",
        "genes": "GSE186538_Human_genes.txt.gz",
        "metadata": "GSE186538_Human_cell_meta.txt.gz",
        "output": "human.h5ad",
    },
    "pig": {
        "source_name": "Pig",
        "matrix": "GSE186538_Pig_counts.mtx.gz",
        "genes": "GSE186538_Pig_genes.txt.gz",
        "metadata": "GSE186538_Pig_cell_meta.txt.gz",
        "output": "pig.h5ad",
    },
    "macaM": {
        "source_name": "Rhesus",
        "matrix": "GSE186538_Rhesus_counts.mtx.gz",
        "genes": "GSE186538_Rhesus_genes.txt.gz",
        "metadata": "GSE186538_Rhesus_cell_meta.txt.gz",
        "output": "macaM.h5ad",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create annotated, QC-filtered h5ad files from the three-species "
            "hippocampus matrices."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument(
        "--species",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Species to process. Missing matrices are skipped by default.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail instead of skipping a species whose input files are missing.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, str]:
    annotations: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["barcode", "cell_type"]:
            raise ValueError(
                f"Expected annotation columns ['barcode', 'cell_type'], "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            barcode = row["barcode"].strip()
            cell_type = row["cell_type"].strip()
            if barcode in annotations:
                raise ValueError(f"Duplicate annotation barcode: {barcode}")
            if barcode and cell_type:
                annotations[barcode] = cell_type
    return annotations


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def read_metadata_barcodes(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "cell_name" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a cell_name column")
        return [row["cell_name"].strip() for row in reader]


def make_unique(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for value in values:
        count = seen.get(value, 0)
        unique.append(value if count == 0 else f"{value}-{count}")
        seen[value] = count + 1
    return unique


def iter_matrix_entries(
    path: Path,
) -> tuple[tuple[int, int, int], object]:
    handle = gzip.open(path, "rt", encoding="ascii")
    try:
        first = handle.readline()
        if not first.startswith("%%MatrixMarket"):
            raise ValueError(f"{path} is not a Matrix Market file")
        line = handle.readline()
        while line.startswith("%"):
            line = handle.readline()
        dimensions = tuple(int(value) for value in line.split())
        if len(dimensions) != 3:
            raise ValueError(f"Invalid Matrix Market dimensions in {path}: {line}")
        return dimensions, handle
    except Exception:
        handle.close()
        raise


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
        else np.zeros(matrix.shape[0], dtype=np.float64)
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
    input_dir: Path,
    output_path: Path,
    annotation_path: Path,
    annotations: dict[str, str],
    min_genes: int,
    min_cells: int,
    max_pct_mt: float,
) -> None:
    import anndata as ad
    import pandas as pd

    matrix_path = input_dir / config["matrix"]
    genes_path = input_dir / config["genes"]
    metadata_path = input_dir / config["metadata"]
    genes = make_unique(read_gzip_lines(genes_path))
    source_barcodes = read_metadata_barcodes(metadata_path)
    annotation_keep = np.asarray(
        [barcode in annotations for barcode in source_barcodes], dtype=bool
    )
    if not annotation_keep.any():
        raise ValueError(f"No annotation barcodes matched {species}")

    mt_mask_all = np.asarray([is_mito_gene(gene) for gene in genes], dtype=bool)
    total_counts_all = np.zeros(len(source_barcodes), dtype=np.float64)
    n_genes_all = np.zeros(len(source_barcodes), dtype=np.int32)
    mt_counts_all = np.zeros(len(source_barcodes), dtype=np.float64)

    dimensions, entries = iter_matrix_entries(matrix_path)
    matrix_genes, matrix_cells, _ = dimensions
    if (matrix_genes, matrix_cells) != (len(genes), len(source_barcodes)):
        entries.close()
        raise ValueError(
            f"Shape mismatch for {matrix_path.name}: "
            f"matrix={(matrix_genes, matrix_cells)}, "
            f"metadata cells={len(source_barcodes)}, genes={len(genes)}"
        )
    try:
        for line in entries:
            gene_text, cell_text, value_text = line.split()
            gene_index = int(gene_text) - 1
            cell_index = int(cell_text) - 1
            if not annotation_keep[cell_index]:
                continue
            value = float(value_text)
            total_counts_all[cell_index] += value
            n_genes_all[cell_index] += 1
            if mt_mask_all[gene_index]:
                mt_counts_all[cell_index] += value
    finally:
        entries.close()

    pct_mt_all = np.divide(
        mt_counts_all * 100.0,
        total_counts_all,
        out=np.zeros_like(total_counts_all),
        where=total_counts_all > 0,
    )
    cell_keep = (
        annotation_keep
        & (n_genes_all >= min_genes)
        & (pct_mt_all <= max_pct_mt)
    )

    gene_n_cells = np.zeros(len(genes), dtype=np.int32)
    _, entries = iter_matrix_entries(matrix_path)
    try:
        for line in entries:
            gene_text, cell_text, _ = line.split()
            cell_index = int(cell_text) - 1
            if cell_keep[cell_index]:
                gene_n_cells[int(gene_text) - 1] += 1
    finally:
        entries.close()
    gene_keep = gene_n_cells >= min_cells
    kept_cell_indices = np.flatnonzero(cell_keep)
    kept_gene_indices = np.flatnonzero(gene_keep)
    cell_map = np.full(len(source_barcodes), -1, dtype=np.int32)
    gene_map = np.full(len(genes), -1, dtype=np.int32)
    cell_map[kept_cell_indices] = np.arange(len(kept_cell_indices), dtype=np.int32)
    gene_map[kept_gene_indices] = np.arange(len(kept_gene_indices), dtype=np.int32)
    max_nnz = int(n_genes_all[cell_keep].sum())

    with tempfile.TemporaryDirectory(prefix=f"hippocampus_{species}_") as temp_dir:
        temp_path = Path(temp_dir)
        data = np.memmap(
            temp_path / "data.bin", mode="w+", dtype=np.float32, shape=max_nnz
        )
        indices = np.memmap(
            temp_path / "indices.bin", mode="w+", dtype=np.int32, shape=max_nnz
        )
        indptr = np.zeros(len(kept_cell_indices) + 1, dtype=np.int64)
        write_position = 0
        previous_source_cell = -1

        _, entries = iter_matrix_entries(matrix_path)
        try:
            for line in entries:
                gene_text, cell_text, value_text = line.split()
                source_cell = int(cell_text) - 1
                if source_cell < previous_source_cell:
                    raise ValueError(
                        f"{matrix_path} entries are not ordered by cell column"
                    )
                previous_source_cell = source_cell
                output_cell = cell_map[source_cell]
                if output_cell < 0:
                    continue
                output_gene = gene_map[int(gene_text) - 1]
                if output_gene < 0:
                    continue
                indices[write_position] = output_gene
                data[write_position] = float(value_text)
                indptr[output_cell + 1] += 1
                write_position += 1
        finally:
            entries.close()

        np.cumsum(indptr, out=indptr)
        matrix = sparse.csr_matrix(
            (data[:write_position], indices[:write_position], indptr),
            shape=(len(kept_cell_indices), len(kept_gene_indices)),
            copy=False,
        )
        barcodes = [source_barcodes[index] for index in kept_cell_indices]
        cell_types = [annotations[barcode] for barcode in barcodes]
        kept_genes = [genes[index] for index in kept_gene_indices]
        mt_mask = np.asarray(
            [is_mito_gene(gene) for gene in kept_genes], dtype=bool
        )
        n_genes = np.diff(indptr)
        total_counts = np.zeros(len(kept_cell_indices), dtype=np.float64)
        mt_counts = np.zeros(len(kept_cell_indices), dtype=np.float64)
        for row_index in range(len(kept_cell_indices)):
            start, end = indptr[row_index : row_index + 2]
            row_data = data[start:end]
            row_genes = indices[start:end]
            total_counts[row_index] = row_data.sum(dtype=np.float64)
            mt_counts[row_index] = row_data[mt_mask[row_genes]].sum(
                dtype=np.float64
            )
        pct_mt = np.divide(
            mt_counts * 100.0,
            total_counts,
            out=np.zeros_like(total_counts),
            where=total_counts > 0,
        )
        obs = pd.DataFrame(
            {
                "cell_type": cell_types,
                "species": species,
                "total_counts": total_counts,
                "n_genes_by_counts": n_genes,
                "pct_counts_mt": pct_mt,
            },
            index=pd.Index(barcodes, name="barcode"),
        )
        var = pd.DataFrame(
            {
                "n_cells_by_counts": gene_n_cells[gene_keep],
                "mt": mt_mask,
            },
            index=pd.Index(kept_genes, name="gene"),
        )
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
        adata.uns["qc_thresholds"] = {
            "min_genes": int(min_genes),
            "min_cells": int(min_cells),
            "max_pct_mt": float(max_pct_mt),
        }
        adata.uns["source_files"] = {
            "matrix": str(matrix_path),
            "genes": str(genes_path),
            "metadata": str(metadata_path),
        }
        adata.uns["annotation_file"] = str(annotation_path)
        adata.write_h5ad(output_path, compression="gzip")

    print(
        f"{species}: source={len(source_barcodes)}, "
        f"annotated={int(annotation_keep.sum())}, "
        f"saved={adata.n_obs} cells x {adata.n_vars} genes -> {output_path}"
    )


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotation_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    for species in args.species:
        config = DATASETS[species]
        required_paths = [
            args.input_dir / config["matrix"],
            args.input_dir / config["genes"],
            args.input_dir / config["metadata"],
        ]
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            message = (
                f"{species}: skipped because input file(s) are missing: "
                + ", ".join(str(path) for path in missing)
            )
            if args.require_all:
                raise FileNotFoundError(message)
            print(message)
            continue

        process_dataset(
            species=species,
            config=config,
            input_dir=args.input_dir,
            output_path=args.output_dir / config["output"],
            annotation_path=args.annotation_csv,
            annotations=annotations,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            max_pct_mt=args.max_pct_mt,
        )
        completed += 1

    if completed == 0:
        raise RuntimeError("No dataset was processed")


if __name__ == "__main__":
    main()
