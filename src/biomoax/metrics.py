"""Cosine retrieval with explicit query eligibility and positive masks."""
import math
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite two-dimensional feature matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Zero profiles must be removed explicitly before retrieval")
    return x / norms


def retrieve(query, gallery, positive, excluded=None, ks=(1, 5, 10)):
    q, g = unit(query), unit(gallery)
    positive = np.asarray(positive)
    if positive.dtype != bool or positive.shape != (len(q), len(g)):
        raise ValueError("positive must be a boolean [queries, gallery] matrix")
    excluded = np.zeros_like(positive) if excluded is None else np.asarray(excluded)
    if excluded.dtype != bool or excluded.shape != positive.shape:
        raise ValueError("excluded must match the positive mask and have bool dtype")
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in ks):
        raise ValueError("K values must be positive integers")
    positive = positive & ~excluded
    eligible = positive.any(axis=1)
    similarities = q @ g.T
    similarities[excluded] = -np.inf
    order = np.argsort(-similarities, axis=1, kind="stable")
    ranked_hits = np.take_along_axis(positive, order, axis=1)
    result = {"n_queries": len(q), "n_eligible": int(eligible.sum()), "gallery_size": len(g)}
    for k in ks:
        hits = ranked_hits[:, :min(k, len(g))].any(axis=1)
        rates = []
        for i in np.flatnonzero(eligible):
            n = int((~excluded[i]).sum()); m = int(positive[i].sum()); take = min(k, n)
            rates.append(1 - (math.comb(n-m, take) / math.comb(n, take) if take <= n-m else 0))
        result[f"top{k}_percent"] = float(hits[eligible].mean()*100) if eligible.any() else None
        result[f"random_top{k}_percent"] = float(np.mean(rates)*100) if rates else None
    return result
