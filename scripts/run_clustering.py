#!/usr/bin/env python3
"""Builds extended feature set and runs UMAP + HDBSCAN clustering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_extraction import extract_simple_features
from src.preprocessing import preprocess_lc

RAW_DEFAULT = Path("notebooks/data/raw/ogle4/2025/photometry")
FEATURES_DEFAULT = Path("notebooks/data/features/features_ogle2025_extended.csv")
PLOT_DEFAULT = Path("notebooks/data/features/umap_clusters.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=RAW_DEFAULT, help="Directory with OGLE photometry .dat files")
    parser.add_argument("--features_csv", type=Path, default=FEATURES_DEFAULT, help="Path to write the feature table")
    parser.add_argument("--plot_path", type=Path, default=PLOT_DEFAULT, help="Path for the clustering scatter plot")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.2, help="UMAP min_dist")
    parser.add_argument("--cluster-min-size", type=int, default=25, help="HDBSCAN min_cluster_size")
    parser.add_argument("--cluster-min-samples", type=int, default=10, help="HDBSCAN min_samples")
    return parser.parse_args()


def load_lightcurve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=["HJD", "mag", "err", "unk1", "unk2"],
        engine="c",
    )
    df["event"] = path.stem.split("_")[0]
    return df


def build_feature_table(raw_dir: Path) -> pd.DataFrame:
    rows = []
    files = sorted(raw_dir.glob("*.dat"))
    for file in tqdm(files, desc="Extracting features"):
        df = preprocess_lc(load_lightcurve(file))
        if df.empty:
            continue
        feats = extract_simple_features(df["t_rel"].values, df["flux"].values, df.get("err").values)
        feats["event"] = df["event"].iloc[0]
        rows.append(feats)
    features = pd.DataFrame(rows)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def cluster_features(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    numeric_cols = sorted([c for c in features.columns if features[c].dtype != object and c != "event"])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = features[numeric_cols].to_numpy()
    X = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
        verbose=True,
    )
    embedding = reducer.fit_transform(X_scaled)
    features["umap_x"] = embedding[:, 0]
    features["umap_y"] = embedding[:, 1]
    print("UMAP embedding complete.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.cluster_min_size,
        min_samples=args.cluster_min_samples,
        cluster_selection_epsilon=0.05,
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(embedding)
    features["cluster"] = labels
    features["cluster_probability"] = clusterer.probabilities_
    print(f"HDBSCAN assigned {len(set(labels))} unique labels.")
    return features


def summarize(features: pd.DataFrame, plot_path: Path) -> None:
    summary = features["cluster"].value_counts(dropna=False).to_dict()
    print("Cluster counts:", json.dumps(summary, indent=2))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features["umap_x"],
        features["umap_y"],
        c=features["cluster"],
        cmap="tab10",
        s=10,
        alpha=0.9,
    )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("OGLE 2025 events – extended features")
    plt.colorbar(scatter, label="cluster")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    features = build_feature_table(args.raw_dir)
    if features.empty:
        raise SystemExit("No features extracted; check raw_dir")
    print(f"Built feature table with {len(features)} events and {features.shape[1]} columns.")

    features = cluster_features(features, args)
    args.features_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_csv, index=False)
    summarize(features, args.plot_path)
    print(f"Saved features to {args.features_csv}")
    print(f"Saved plot to {args.plot_path}")


if __name__ == "__main__":
    main()
