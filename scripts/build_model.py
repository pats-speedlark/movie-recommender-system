"""Build the lightweight recommendation artifacts used by app.py.

The original app shipped a dense 4806x4806 float64 similarity matrix
(~185 MB, stored in Git LFS). Loading it was slow, used a lot of memory and
broke deployments where LFS files are not fetched (the app crashed on
start-up trying to unpickle an LFS pointer file).

This script replaces it with:
  * data/movies.csv       - movie_id, title, tags
  * data/neighbors.npz    - top-K neighbour indices + scores per movie (~1 MB)

Usage (dev only, needs scikit-learn):
    pip install scikit-learn pandas numpy
    python scripts/build_model.py --source movie_dict.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TOP_K = 50


def load_source(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        with open(path, "rb") as fh:
            df = pd.DataFrame(pickle.load(fh))
    df = df[["movie_id", "title", "tags"]].dropna(subset=["title"])
    df["tags"] = df["tags"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
    return df


def build(df: pd.DataFrame, top_k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    # TF-IDF down-weights very common tokens (e.g. "drama", "film") so rare,
    # descriptive signals like cast, director and specific keywords matter
    # more than with the plain CountVectorizer used originally.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
        sublinear_tf=True,
        max_features=50_000,
    )
    matrix = vectorizer.fit_transform(df["tags"])
    sim = linear_kernel(matrix, matrix).astype(np.float32)  # rows are L2-normalised -> cosine
    np.fill_diagonal(sim, -1.0)

    k = min(top_k, len(df) - 1)
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    scores = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-scores, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    return idx.astype(np.int32), scores.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DATA_DIR / "movies.csv"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    df = load_source(Path(args.source))
    idx, scores = build(df, args.top_k)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "movies.csv", index=False)
    np.savez_compressed(DATA_DIR / "neighbors.npz", indices=idx, scores=scores)
    print(f"Built neighbours for {len(df)} movies (top {idx.shape[1]}) -> {DATA_DIR}")


if __name__ == "__main__":
    main()
