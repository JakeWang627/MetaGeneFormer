from pathlib import Path
import subprocess
import sys

import pandas as pd


VIGNETTE_DIR = Path(__file__).resolve().parent
METAGENEFORMER_DIR = VIGNETTE_DIR.parents[1]
SMOKE_DIR = VIGNETTE_DIR / "smoke_test_copy_one_epoch"
RUN_CSV = SMOKE_DIR / "3_species_run.csv"
MODEL_PATH = SMOKE_DIR / "gastric_pretrain_model_seed_0.pt"

df = pd.DataFrame(
    {
        "path": [
            "D:/111icde_addition_experiments/3_species_gastric/processed_h5ad/human.h5ad",
            "D:/111icde_addition_experiments/3_species_gastric/processed_h5ad/mouse.h5ad",
            "D:/111icde_addition_experiments/3_species_gastric/processed_h5ad/pig.h5ad",
        ],
        "species": ["human", "mouse", "pig"],
        "embedding_path": [
            "D:/protein_go_ontology_embeddings/human_protein_go_fused.fused_embedding.pkl.gz",
            "D:/protein_go_ontology_embeddings/mouse_protein_go_fused.fused_embedding.pkl.gz",
            "D:/protein_go_ontology_embeddings/pig_protein_go_fused.fused_embedding.pkl.gz",
        ],
    }
)

for column in ("path", "embedding_path"):
    missing = [path for path in df[column] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {column} files: {missing}")

SMOKE_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RUN_CSV, index=False)
print(f"Saved run CSV: {RUN_CSV}", flush=True)

command = [
    sys.executable,
    "train-saturn.py",
    f"--in_data={RUN_CSV}",
    "--device_num=0",
    "--in_label_col=cell_type",
    "--ref_label_col=cell_type",
    f"--work_dir={SMOKE_DIR.as_posix()}/",
    "--num_macrogenes=2000",
    "--pretrain",
    "--model_dim=256",
    "--pretrain_batch_size=256",
    "--hv_genes=8000",
    "--pretrain_epochs=1",
    "--pe_sim_penalty=1.0",
    "--l1_penalty=0",
    "--centroid_score_func=default",
    "--embedding_model=ESM2",
    "--seed=0",
    "--org=3_species_gastric_copy_smoke_test",
    f"--pretrain_model_path={MODEL_PATH}",
    f"--centroids_init_path={SMOKE_DIR / 'centroids_seed_0'}",
]

print("Running:", " ".join(map(str, command)), flush=True)
subprocess.run(command, cwd=METAGENEFORMER_DIR, check=True)

pretrain_files = sorted((SMOKE_DIR / "saturn_results").glob("*_pretrain.h5ad"))
if not MODEL_PATH.exists() or not pretrain_files:
    raise RuntimeError("Pretraining finished without the expected outputs")

print(f"Pretrain model: {MODEL_PATH}", flush=True)
print(f"Pretrain AnnData: {pretrain_files[-1]}", flush=True)
