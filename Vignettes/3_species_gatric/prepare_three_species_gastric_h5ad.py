#!/usr/bin/env python
"""Convert three gastric expression matrices to annotated, QC-filtered h5ad files."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "data"
DEFAULT_ANNOTATION = (
    SCRIPT_DIR / "data" / "three_species_gastric_antrum_barcode_cell_type.csv"
)

DATASETS = {
    "human": "GSE225275_human_data.txt.gz",
    "mouse": "GSE225275_mouse_data.txt.gz",
    "pig": "GSE225275_pig_data.txt.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create QC-filtered h5ad files from the three gastric expression matrices."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "processed_h5ad",
    )
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, str]:
    annotations: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["barcode", "cell_type"]:
            raise ValueError(
                f"Expected annotation columns ['barcode', 'cell_type'], got {reader.fieldnames}"
            )
        for row in reader:
            barcode = row["barcode"]
            if barcode in annotations:
                raise ValueError(f"Duplicate annotation barcode: {barcode}")
            annotations[barcode] = row["cell_type"]
    return annotations


def make_unique(values: list[str]) -> list[str]:
    totals: dict[str, int] = {}
    result: list[str] = []
    for value in values:
        count = totals.get(value, 0)
        result.append(value if count == 0 else f"{value}-{count}")
        totals[value] = count + 1
    return result


def is_mito_gene(gene: str) -> bool:
    upper = gene.upper()
    return upper.startswith("MT-") or upper.startswith("MT.")


def read_annotated_matrix(
    path: Path,
    annotations: dict[str, str],
) -> tuple[sparse.csr_matrix, list[str], list[str], list[str]]:
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    genes: list[str] = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        source_barcodes = header[1:]
        selected_source_columns = np.asarray(
            [i for i, barcode in enumerate(source_barcodes) if barcode in annotations],
            dtype=np.int64,
        )
        barcodes = [source_barcodes[i] for i in selected_source_columns]
        cell_types = [annotations[barcode] for barcode in barcodes]

        if not barcodes:
            raise ValueError(f"No annotated barcodes matched {path}")

        expected_values = len(source_barcodes)
        for gene_index, line in enumerate(handle):
            gene, separator, values_text = line.rstrip("\r\n").partition("\t")
            if not separator:
                raise ValueError(f"Malformed line {gene_index + 2} in {path}")
            values = np.fromstring(values_text, sep="\t", dtype=np.float32)
            if values.size != expected_values:
                raise ValueError(
                    f"{path.name}, line {gene_index + 2}: expected {expected_values} "
                    f"values, found {values.size}"
                )
            selected = values[selected_source_columns]
            nonzero_cells = np.flatnonzero(selected)
            if nonzero_cells.size:
                row_parts.append(nonzero_cells.astype(np.int32, copy=False))
                col_parts.append(
                    np.full(nonzero_cells.size, gene_index, dtype=np.int32)
                )
                data_parts.append(selected[nonzero_cells])
            genes.append(gene)

    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int32)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(barcodes), len(genes)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, barcodes, cell_types, make_unique(genes)


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
        out=np.zeros_like(mt_counts, dtype=np.float32),
        where=total_counts > 0,
    )
    return total_counts, n_genes, pct_mt, mt_mask


def process_dataset(
    species: str,
    input_path: Path,
    output_path: Path,
    annotations: dict[str, str],
    annotation_path: Path,
    min_genes: int,
    min_cells: int,
    max_pct_mt: float,
) -> None:
    import anndata as ad
    import pandas as pd

    matrix, barcodes, cell_types, genes = read_annotated_matrix(
        input_path, annotations
    )
    input_cells, input_genes = matrix.shape

    total_counts, n_genes, pct_mt, _ = calculate_qc(matrix, genes)
    cell_keep = (n_genes >= min_genes) & (pct_mt <= max_pct_mt)
    matrix = matrix[cell_keep].tocsr()
    barcodes = [barcode for barcode, keep in zip(barcodes, cell_keep) if keep]
    cell_types = [
        cell_type for cell_type, keep in zip(cell_types, cell_keep) if keep
    ]

    gene_n_cells = np.asarray((matrix > 0).sum(axis=0)).ravel()
    gene_keep = gene_n_cells >= min_cells
    matrix = matrix[:, gene_keep].tocsr()
    genes = [gene for gene, keep in zip(genes, gene_keep) if keep]

    total_counts, n_genes, pct_mt, mt_mask = calculate_qc(matrix, genes)
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
    adata.uns["source_file"] = str(input_path)
    adata.uns["annotation_file"] = str(annotation_path)
    adata.write_h5ad(output_path, compression="gzip")

    print(
        f"{species}: matched {input_cells} annotated cells and {input_genes} genes; "
        f"saved {adata.n_obs} cells x {adata.n_vars} genes -> {output_path}"
    )


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotation_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for species, filename in DATASETS.items():
        input_path = args.input_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        output_path = args.output_dir / f"{species}.h5ad"
        process_dataset(
            species=species,
            input_path=input_path,
            output_path=output_path,
            annotations=annotations,
            annotation_path=args.annotation_csv,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            max_pct_mt=args.max_pct_mt,
        )


if __name__ == "__main__":
    main()
