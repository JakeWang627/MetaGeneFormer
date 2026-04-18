#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import argparse
import random
from collections import defaultdict

import numpy as np
import psutil
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def ensure_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def choose_device(device_arg: str):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_intra_species_knn(X, species_labels, k=15):
    N = X.shape[0]
    neighbors_intra = [[] for _ in range(N)]
    sp2idx = defaultdict(list)
    for i, sp in enumerate(species_labels):
        sp2idx[int(sp)].append(i)

    for _, idxs in sp2idx.items():
        if len(idxs) <= 1:
            continue
        Xs = X[idxs]
        kk = min(k + 1, len(idxs))
        nn_model = NearestNeighbors(n_neighbors=kk, metric="euclidean", n_jobs=-1).fit(Xs)
        _, nbrs = nn_model.kneighbors(Xs, return_distance=True)
        for local_i, neigh in enumerate(nbrs):
            gi = idxs[local_i]
            neighbors_intra[gi] = [idxs[j] for j in neigh[1:]]
    return neighbors_intra


def build_cross_species_mnn_by_label(X, train_labels, species_labels, k=5):
    N = X.shape[0]
    neighbors_cross = [[] for _ in range(N)]

    valid_mask = train_labels >= 0
    unique_labels = np.unique(train_labels[valid_mask])
    unique_species = np.unique(species_labels)

    for lab in unique_labels:
        idx_lab = np.where(train_labels == lab)[0]
        if len(idx_lab) < 2:
            continue

        sp2idx = {}
        for sp in unique_species:
            idx = idx_lab[species_labels[idx_lab] == sp]
            if len(idx) > 0:
                sp2idx[int(sp)] = idx

        sps = list(sp2idx.keys())
        if len(sps) < 2:
            continue

        for i in range(len(sps)):
            for j in range(i + 1, len(sps)):
                sa, sb = sps[i], sps[j]
                A = sp2idx[sa]
                B = sp2idx[sb]
                if len(A) == 0 or len(B) == 0:
                    continue

                kA = min(k, len(B))
                kB = min(k, len(A))

                nn_AB = NearestNeighbors(n_neighbors=kA, metric="euclidean", n_jobs=-1).fit(X[B])
                idx_AB = nn_AB.kneighbors(X[A], return_distance=False)

                nn_BA = NearestNeighbors(n_neighbors=kB, metric="euclidean", n_jobs=-1).fit(X[A])
                idx_BA = nn_BA.kneighbors(X[B], return_distance=False)

                A_to_B = [set(B[idx_AB[t]]) for t in range(len(A))]
                B_to_A = [set(A[idx_BA[t]]) for t in range(len(B))]
                B_to_A_map = {B[t]: B_to_A[t] for t in range(len(B))}

                for t, a in enumerate(A):
                    for b in A_to_B[t]:
                        if a in B_to_A_map.get(int(b), set()):
                            neighbors_cross[int(a)].append(int(b))
                            neighbors_cross[int(b)].append(int(a))

    neighbors_cross = [sorted(list(set(nbs))) for nbs in neighbors_cross]
    return neighbors_cross


class IdxDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def make_subgraph_collate(features, labels_train, species_labels,
                          neighbors_intra, neighbors_cross,
                          n_expand_intra=15, n_expand_cross=5):
    features = np.asarray(features)
    labels_train = np.asarray(labels_train)
    species_labels = np.asarray(species_labels)

    def collate(batch_idx_list):
        seed_idx = np.array(batch_idx_list, dtype=np.int64)

        sub_set = set(seed_idx.tolist())
        for g in seed_idx:
            for nb in neighbors_intra[g][:n_expand_intra]:
                sub_set.add(nb)
            for nb in neighbors_cross[g][:n_expand_cross]:
                sub_set.add(nb)

        sub_nodes = np.array(sorted(list(sub_set)), dtype=np.int64)
        pos = {g: i for i, g in enumerate(sub_nodes)}
        seed_pos = np.array([pos[g] for g in seed_idx], dtype=np.int64)

        rows, cols, etype = [], [], []
        for g in sub_nodes:
            src = pos[g]
            for nb in neighbors_intra[g][:n_expand_intra]:
                if nb in pos:
                    rows.append(src)
                    cols.append(pos[nb])
                    etype.append(0)
            for nb in neighbors_cross[g][:n_expand_cross]:
                if nb in pos:
                    rows.append(src)
                    cols.append(pos[nb])
                    etype.append(1)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)

        x_sub = torch.tensor(features[sub_nodes], dtype=torch.float32)
        y_seed_train = torch.tensor(labels_train[seed_idx], dtype=torch.long)
        sp_seed = torch.tensor(species_labels[seed_idx], dtype=torch.long)

        return x_sub, y_seed_train, sp_seed, edge_index, edge_type, torch.tensor(seed_pos, dtype=torch.long)

    return collate


class SoftTripleLoss(nn.Module):
    def __init__(self, embedding_dim, n_classes, n_centers=5, la=10.0, gamma=0.1):
        super().__init__()
        self.la = la
        self.gamma = gamma
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.centers = nn.Parameter(torch.randn(n_classes * n_centers, embedding_dim) / np.sqrt(embedding_dim))
        self.weight = nn.Parameter(torch.ones(n_classes, n_centers) / n_centers)

    def forward(self, x, labels):
        x = F.normalize(x, dim=1)
        centers = F.normalize(self.centers, dim=1)
        sim = torch.matmul(x, centers.t())
        B = x.size(0)
        sim = sim.view(B, self.n_classes, self.n_centers)
        weight = F.softmax(self.weight, dim=1)
        sim_weighted = torch.sum(sim * weight, dim=2)
        mask = torch.zeros_like(sim_weighted).scatter_(1, labels.unsqueeze(1), 1)
        sim_target = torch.sum(sim_weighted * mask, dim=1)
        sim_others = torch.max(sim_weighted * (1 - mask), dim=1)[0]
        loss = torch.mean(F.softplus(self.la * (sim_others - sim_target + self.gamma)))
        return loss


def variance_loss(embeddings, labels):
    unique_labels = labels.unique()
    loss = 0.0
    count = 0
    for lab in unique_labels:
        mask = labels == lab
        cluster = embeddings[mask]
        if cluster.size(0) < 2:
            continue
        center = cluster.mean(dim=0, keepdim=True)
        loss += ((cluster - center) ** 2).sum(dim=1).mean()
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss / count


def graph_smoothness_loss(z, edge_index, edge_type, alpha=None):
    src = edge_index[0]
    dst = edge_index[1]
    mask = edge_type == 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    src = src[mask]
    dst = dst[mask]
    diff2 = (z[src] - z[dst]).pow(2).sum(dim=1)
    if alpha is None:
        return diff2.mean()
    a = alpha[mask]
    return (a * diff2).mean()


def mnn_pull_loss(z, edge_index, edge_type, alpha=None):
    src = edge_index[0]
    dst = edge_index[1]
    mask = edge_type == 1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    src = src[mask]
    dst = dst[mask]
    diff2 = (z[src] - z[dst]).pow(2).sum(dim=1)
    if alpha is None:
        return diff2.mean()
    a = alpha[mask]
    return (a * diff2).mean()


class FixedEdgeWeightGNN(nn.Module):
    def __init__(self, dim, edge_type_emb_dim=8, hidden=128, dropout=0.1):
        super().__init__()
        self.type_emb = nn.Embedding(2, edge_type_emb_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 3 + edge_type_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, z, edge_index, edge_type):
        src = edge_index[0]
        dst = edge_index[1]
        zi = z[src]
        zj = z[dst]
        te = self.type_emb(edge_type)
        e_in = torch.cat([zi, zj, (zi - zj).abs(), te], dim=1)
        w = torch.sigmoid(self.edge_mlp(e_in)).squeeze(1)

        N = z.size(0)
        max_per_src = torch.full((N,), -1e9, device=z.device)
        max_per_src.scatter_reduce_(0, src, w, reduce="amax", include_self=True)
        w_exp = torch.exp(w - max_per_src[src])
        sum_per_src = torch.zeros((N,), device=z.device)
        sum_per_src.scatter_add_(0, src, w_exp)
        alpha = w_exp / (sum_per_src[src] + 1e-12)

        msg = self.msg_mlp(zj) * alpha.unsqueeze(1)
        agg = torch.zeros_like(z)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, z.size(1)), msg)

        out = self.norm(z + F.dropout(agg, p=self.dropout, training=self.training))
        return out, alpha


class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, num_heads=4, num_layers=1, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.2
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.gnn = FixedEdgeWeightGNN(dim=embedding_dim, dropout=0.1)

    def forward(self, x_sub, edge_index, edge_type):
        h = self.input_proj(x_sub)
        h = h.unsqueeze(0)
        h = self.encoder(h).squeeze(0)
        z = self.out_proj(h)
        z = F.normalize(z, dim=1)
        z2, alpha = self.gnn(z, edge_index, edge_type)
        z2 = F.normalize(z2, dim=1)
        return z2, alpha


def infer_all_embeddings(model, features,
                         neighbors_intra, neighbors_cross,
                         batch_size=1024, n_expand_intra=15, n_expand_cross=5,
                         device="cuda"):
    model.eval()
    N = features.shape[0]
    D = model.out_proj.out_features
    Z = np.zeros((N, D), dtype=np.float32)

    features = np.asarray(features)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            seed_idx = np.arange(start, min(start + batch_size, N), dtype=np.int64)

            sub_set = set(seed_idx.tolist())
            for g in seed_idx:
                for nb in neighbors_intra[g][:n_expand_intra]:
                    sub_set.add(nb)
                for nb in neighbors_cross[g][:n_expand_cross]:
                    sub_set.add(nb)

            sub_nodes = np.array(sorted(list(sub_set)), dtype=np.int64)
            pos = {g: i for i, g in enumerate(sub_nodes)}
            seed_pos = np.array([pos[g] for g in seed_idx], dtype=np.int64)

            rows, cols, etype = [], [], []
            for g in sub_nodes:
                src = pos[g]
                for nb in neighbors_intra[g][:n_expand_intra]:
                    if nb in pos:
                        rows.append(src)
                        cols.append(pos[nb])
                        etype.append(0)
                for nb in neighbors_cross[g][:n_expand_cross]:
                    if nb in pos:
                        rows.append(src)
                        cols.append(pos[nb])
                        etype.append(1)

            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
            edge_type = torch.tensor(etype, dtype=torch.long, device=device)
            x_sub = torch.tensor(features[sub_nodes], dtype=torch.float32, device=device)
            seed_pos_t = torch.tensor(seed_pos, dtype=torch.long, device=device)

            z_sub, _ = model(x_sub, edge_index, edge_type)
            Z[seed_idx] = z_sub[seed_pos_t].cpu().numpy()

    return Z


def make_partial_label_mask_stratified(labels_full, species_labels, label_fraction, seed=2025):
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels_full), dtype=bool)

    groups = defaultdict(list)
    for i, (lab, sp) in enumerate(zip(labels_full, species_labels)):
        groups[(int(sp), int(lab))].append(i)

    for _, idxs in groups.items():
        idxs = np.array(idxs, dtype=np.int64)
        if label_fraction >= 1.0:
            mask[idxs] = True
            continue
        n_keep = max(1, int(round(len(idxs) * label_fraction)))
        chosen = rng.choice(idxs, size=n_keep, replace=False)
        mask[chosen] = True

    return mask


def train_one_fraction(adata, features, labels_full, species_labels, label_names, label_fraction, outdir, args):
    frac_tag = f"{label_fraction:.2f}".rstrip("0").rstrip(".")
    run_dir = os.path.join(outdir, f"fraction_{sanitize_name(frac_tag)}")
    os.makedirs(run_dir, exist_ok=True)

    labeled_mask = make_partial_label_mask_stratified(
        labels_full, species_labels, label_fraction=label_fraction, seed=args.seed
    )
    labels_train = labels_full.copy()
    labels_train[~labeled_mask] = -1

    pca = PCA(n_components=min(50, features.shape[1]), random_state=args.seed)
    X_graph = pca.fit_transform(features).astype(np.float32)

    neighbors_intra = build_intra_species_knn(X_graph, species_labels, k=args.k_intra)
    neighbors_cross = build_cross_species_mnn_by_label(X_graph, labels_train, species_labels, k=args.k_mnn)

    dataset = IdxDataset(len(features))
    collate_fn = make_subgraph_collate(
        features, labels_train, species_labels,
        neighbors_intra, neighbors_cross,
        n_expand_intra=args.n_expand_intra,
        n_expand_cross=args.n_expand_cross,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)
    model = EmbeddingModel(
        input_dim=features.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    criterion = SoftTripleLoss(
        embedding_dim=args.embedding_dim,
        n_classes=len(label_names),
        n_centers=args.n_centers,
    ).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)

    process = psutil.Process(os.getpid())
    train_start_time = time.perf_counter()
    peak_rss_mb = process.memory_info().rss / 1024**2

    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        running = 0.0

        for x_sub, y_seed_train, _, edge_index, edge_type, seed_pos in dataloader:
            x_sub = x_sub.to(device, non_blocking=True)
            y_seed_train = y_seed_train.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            edge_type = edge_type.to(device, non_blocking=True)
            seed_pos = seed_pos.to(device, non_blocking=True)

            z_sub, alpha = model(x_sub, edge_index, edge_type)
            z_seed = z_sub[seed_pos]

            sup_mask = y_seed_train >= 0
            if sup_mask.sum() > 0:
                z_seed_sup = z_seed[sup_mask]
                y_seed_sup = y_seed_train[sup_mask]
                loss_st = criterion(z_seed_sup, y_seed_sup)
                loss_v = variance_loss(z_seed_sup, y_seed_sup)
            else:
                loss_st = torch.tensor(0.0, device=device)
                loss_v = torch.tensor(0.0, device=device)

            loss_g = graph_smoothness_loss(z_sub, edge_index, edge_type, alpha=alpha)
            loss_p = mnn_pull_loss(z_sub, edge_index, edge_type, alpha=alpha)
            loss_w = alpha.mean() if alpha.numel() > 0 else torch.tensor(0.0, device=device)

            loss = (
                loss_st
                + args.lambda_var * loss_v
                + args.lambda_graph * loss_g
                + args.lambda_mnn * loss_p
                + args.lambda_wreg * loss_w
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * x_sub.size(0)

            current_rss_mb = process.memory_info().rss / 1024**2
            peak_rss_mb = max(peak_rss_mb, current_rss_mb)

        if use_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start_time
        epoch_loss = running / len(dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f}")

    if use_cuda:
        torch.cuda.synchronize(device)
    total_train_time = time.perf_counter() - train_start_time

    if use_cuda:
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        current_gpu_mem_mb = torch.cuda.memory_allocated(device) / 1024**2
    else:
        peak_gpu_mem_mb = 0.0
        current_gpu_mem_mb = 0.0

    runtime_stats = {
        "label_fraction": label_fraction,
        "train_time_sec_total": total_train_time,
        "train_time_min_total": total_train_time / 60.0,
        "peak_rss_mb": peak_rss_mb,
        "current_rss_mb": process.memory_info().rss / 1024**2,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "current_gpu_mem_mb": current_gpu_mem_mb,
        "device": str(device),
        "epochs": args.epochs,
        "n_labeled_cells": int(labeled_mask.sum()),
        "n_total_cells": int(len(labels_full)),
    }
    with open(os.path.join(run_dir, "training_runtime_stats.json"), "w", encoding="utf-8") as f:
        json.dump(runtime_stats, f, indent=2, ensure_ascii=False)

    print("\nInfer embeddings...")
    all_embeddings = infer_all_embeddings(
        model, features, neighbors_intra, neighbors_cross,
        batch_size=args.batch_size,
        n_expand_intra=args.n_expand_intra,
        n_expand_cross=args.n_expand_cross,
        device=device,
    )

    ad_out = adata.copy()
    ad_out.obsm[args.output_embedding_key] = all_embeddings

    print("Computing neighbors / UMAP / Leiden...")
    sc.pp.neighbors(ad_out, use_rep=args.output_embedding_key, n_neighbors=args.umap_neighbors, random_state=args.seed)
    sc.tl.umap(ad_out, min_dist=args.umap_min_dist, random_state=args.seed)
    sc.tl.leiden(ad_out, resolution=args.leiden_resolution, key_added=args.leiden_key, random_state=args.seed)

    out_h5ad = os.path.join(run_dir, f"{args.output_prefix}_fraction_{sanitize_name(frac_tag)}.h5ad")
    ad_out.write_h5ad(out_h5ad)
    print(f"Saved h5ad: {out_h5ad}")

    return out_h5ad


def parse_args():
    parser = argparse.ArgumentParser(description="MetaGeneFormer training and saving h5ad with UMAP.")
    parser.add_argument("--input_h5ad", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--species_key", type=str, default="species")
    parser.add_argument("--label_key", type=str, default="labels2")

    parser.add_argument("--embedding-dim", dest="embedding_dim", type=int, default=256)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=4)
    parser.add_argument("--num-layers", dest="num_layers", type=int, default=1)

    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-centers", dest="n_centers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0)

    parser.add_argument("--k-intra", dest="k_intra", type=int, default=15)
    parser.add_argument("--k-mnn", dest="k_mnn", type=int, default=5)
    parser.add_argument("--n-expand-intra", dest="n_expand_intra", type=int, default=15)
    parser.add_argument("--n-expand-cross", dest="n_expand_cross", type=int, default=5)

    parser.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.2)
    parser.add_argument("--lambda-graph", dest="lambda_graph", type=float, default=0.1)
    parser.add_argument("--lambda-mnn", dest="lambda_mnn", type=float, default=0.2)
    parser.add_argument("--lambda-wreg", dest="lambda_wreg", type=float, default=0.01)

    parser.add_argument("--label-fraction", dest="label_fraction", type=float, default=None)
    parser.add_argument("--sweep", nargs="+", type=float, default=None)

    parser.add_argument("--output_embedding_key", type=str, default="X_embed_transformer")
    parser.add_argument("--output_prefix", type=str, default="metageneformer")
    parser.add_argument("--umap_neighbors", type=int, default=12)
    parser.add_argument("--umap_min_dist", type=float, default=0.7)
    parser.add_argument("--leiden-resolution", dest="leiden_resolution", type=float, default=1.0)
    parser.add_argument("--leiden_key", type=str, default="leiden_transformer")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    if args.sweep is not None and len(args.sweep) > 0:
        fractions = args.sweep
    elif args.label_fraction is not None:
        fractions = [args.label_fraction]
    else:
        fractions = [1.0]

    fractions = [float(x) for x in fractions]
    for frac in fractions:
        if not (0 < frac <= 1.0):
            raise ValueError(f"Invalid label fraction: {frac}")

    print("Loading data...")
    adata = sc.read(args.input_h5ad)

    if args.label_key not in adata.obs.columns:
        raise ValueError(f"{args.label_key} not found in adata.obs")
    if args.species_key not in adata.obs.columns:
        raise ValueError(f"{args.species_key} not found in adata.obs")

    keep = ~adata.obs[args.label_key].isna() & ~adata.obs[args.species_key].isna()
    adata = adata[keep].copy()

    label_encoder = LabelEncoder()
    adata.obs[f"{args.label_key}_encoded_full"] = label_encoder.fit_transform(adata.obs[args.label_key].astype(str))
    labels_full = adata.obs[f"{args.label_key}_encoded_full"].values.astype(np.int64)
    label_names = label_encoder.classes_

    species_encoder = LabelEncoder()
    adata.obs[f"{args.species_key}_encoded"] = species_encoder.fit_transform(adata.obs[args.species_key].astype(str))
    species_labels = adata.obs[f"{args.species_key}_encoded"].values.astype(np.int64)

    sc.pp.scale(adata)
    features = ensure_dense(adata.X).astype(np.float32)

    for frac in fractions:
        train_one_fraction(
            adata=adata,
            features=features,
            labels_full=labels_full,
            species_labels=species_labels,
            label_names=label_names,
            label_fraction=frac,
            outdir=args.outdir,
            args=args,
        )

    print("Finished.")


if __name__ == "__main__":
    main()
    

