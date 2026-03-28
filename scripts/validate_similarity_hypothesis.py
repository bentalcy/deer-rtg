"""
Hypothesis validation: can DINOv2 cosine similarity discriminate individual deer?

Uses manually-reviewed cluster assignments from unified_round5 as "same deer" proxy.
All comparisons are side-stratified (left-left or right-right only — never cross-side).

Pass/fail thresholds:
  Rank-1 NN accuracy : pass >80%, marginal 60-80%, fail <60%
  AUC-ROC            : pass >0.85, marginal 0.70-0.85, fail <0.70
  d-prime            : pass >1.5,  marginal 0.5-1.5,   fail <0.5
  FAR at 95% TAR     : pass <15%,  marginal 15-40%,    fail >40%
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_EMB_PT  = Path("data/reid/unified_round5/embeddings_recognizable_plus_manual_round2.pt")
DEFAULT_IDX_CSV = Path("data/reid/unified_round5/index_recognizable_plus_manual_round2.csv")
DEFAULT_CLU_CSV = Path("data/reid/unified_round5/clusters_recognizable_plus_manual_round2_k40_corrected_web.csv")
DEFAULT_OUT_DIR = Path("data/reid")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def verdict(value: float, pass_: float, marginal: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= pass_:
            return "PASS"
        if value >= marginal:
            return "MARGINAL"
        return "FAIL"
    else:
        if value <= pass_:
            return "PASS"
        if value <= marginal:
            return "MARGINAL"
        return "FAIL"


def compute_metrics(
    emb: np.ndarray,
    labels: np.ndarray,
    side: str,
) -> dict:
    """
    emb    : (N, 768) L2-normalized embeddings
    labels : (N,) integer cluster IDs
    side   : label for reporting
    """
    N = len(emb)
    assert len(labels) == N

    # --- pairwise cosine similarity (dot product of normalized vectors) ---
    sims = emb @ emb.T  # (N, N)

    same_mask = labels[:, None] == labels[None, :]  # (N, N)
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)

    same_sims = sims[same_mask]
    diff_sims = sims[diff_mask]

    if len(same_sims) == 0:
        print(f"  [{side}] No same-cluster pairs — skipping.")
        return {}

    # --- 1. Rank-1 NN accuracy ---
    rank1_correct = 0
    for i in range(N):
        row = sims[i].copy()
        row[i] = -2.0  # exclude self
        nn_idx = int(np.argmax(row))
        if labels[nn_idx] == labels[i]:
            rank1_correct += 1
    rank1_acc = rank1_correct / N * 100

    # --- 2. AUC-ROC ---
    # upper triangle only to avoid duplicate pairs
    triu = np.triu_indices(N, k=1)
    pair_sims   = sims[triu]
    pair_labels = same_mask[triu].astype(int)
    auc = roc_auc_score(pair_labels, pair_sims)

    # --- 3. d-prime ---
    mu_same, sigma_same = same_sims.mean(), same_sims.std()
    mu_diff, sigma_diff = diff_sims.mean(), diff_sims.std()
    d_prime = (mu_same - mu_diff) / np.sqrt((sigma_same**2 + sigma_diff**2) / 2 + 1e-9)

    # --- 4. FAR at 95% TAR ---
    fpr, tpr, thresholds = roc_curve(pair_labels, pair_sims)
    idx_95 = np.searchsorted(tpr, 0.95)
    idx_95 = min(idx_95, len(fpr) - 1)
    far_at_95tar = fpr[idx_95] * 100
    tar_achieved = tpr[idx_95] * 100

    return {
        "side": side,
        "n": N,
        "n_clusters": len(np.unique(labels)),
        "n_same_pairs": int(len(same_sims)),
        "n_diff_pairs": int(len(diff_sims)),
        "mu_same": float(mu_same),
        "mu_diff": float(mu_diff),
        "sigma_same": float(sigma_same),
        "sigma_diff": float(sigma_diff),
        "rank1_acc": float(rank1_acc),
        "auc": float(auc),
        "d_prime": float(d_prime),
        "far_at_95tar": float(far_at_95tar),
        "tar_achieved": float(tar_achieved),
        "same_sims": same_sims,
        "diff_sims": diff_sims,
        "fpr": fpr,
        "tpr": tpr,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics(m: dict) -> str:
    side = m["side"]
    r1  = verdict(m["rank1_acc"],  80,  60)
    auc = verdict(m["auc"],        0.85, 0.70)
    dp  = verdict(m["d_prime"],    1.5,  0.5)
    far = verdict(m["far_at_95tar"], 15, 40, higher_is_better=False)

    overall = "PASS" if all(v == "PASS" for v in [r1, auc, dp, far]) else \
              "FAIL" if any(v == "FAIL" for v in [r1, auc, dp, far]) else "MARGINAL"

    lines = [
        f"\n{'='*60}",
        f"  Side: {side.upper()}  |  N={m['n']}  |  {m['n_clusters']} clusters",
        f"  Same-cluster pairs: {m['n_same_pairs']:,}  |  Cross-cluster pairs: {m['n_diff_pairs']:,}",
        f"  Mean similarity  — same: {m['mu_same']:.4f} (±{m['sigma_same']:.4f})"
        f"  diff: {m['mu_diff']:.4f} (±{m['sigma_diff']:.4f})",
        f"{'='*60}",
        f"  Rank-1 NN accuracy : {m['rank1_acc']:5.1f}%   [{r1}]  (pass >80%)",
        f"  AUC-ROC            : {m['auc']:.4f}    [{auc}]  (pass >0.85)",
        f"  d-prime            : {m['d_prime']:.4f}    [{dp}]  (pass >1.5)",
        f"  FAR at ~{m['tar_achieved']:.0f}% TAR   : {m['far_at_95tar']:5.1f}%   [{far}]  (pass <15%)",
        f"{'='*60}",
        f"  OVERALL: {overall}",
        f"{'='*60}",
    ]
    print("\n".join(lines))
    return overall


def print_caveats() -> None:
    print("""
CAVEATS (read before acting on results):
  1. Inflated positives: same-cluster images may be visually similar due to
     scene/pose overlap (burst frames). Similarity may reflect scene, not deer
     identity. Real-world accuracy on diverse images will likely be lower.
  2. Noisy negatives: different clusters may be the same deer (cluster errors).
     This deflates cross-cluster similarity, making separation look better.
  3. Both biases favor passing. Marginal results likely mean failure in practice.
     Strong results (d'>2, AUC>0.95) likely reflect real signal.
  4. This dataset is the target deer population (ranger images). A pass here
     is meaningful — but validate again once real enrollment+identification is live.
""")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_histograms(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, m in zip(axes, results):
        bins = np.linspace(-0.2, 1.0, 60)
        ax.hist(m["same_sims"], bins=bins, alpha=0.6, color="green",
                label=f"Same deer (n={len(m['same_sims']):,})", density=True)
        ax.hist(m["diff_sims"], bins=bins, alpha=0.6, color="red",
                label=f"Diff deer (n={len(m['diff_sims']):,})", density=True)
        ax.axvline(m["mu_same"], color="green", linestyle="--", linewidth=1.5)
        ax.axvline(m["mu_diff"], color="red",   linestyle="--", linewidth=1.5)
        ax.set_title(f"Cosine similarity distribution — {m['side']}\n"
                     f"d'={m['d_prime']:.2f}  AUC={m['auc']:.3f}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved: {out_path}")


def plot_roc(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for m in results:
        ax.plot(m["fpr"], m["tpr"],
                label=f"{m['side']} (AUC={m['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title("ROC — DINOv2 similarity (same-side pairs only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb",  default=str(DEFAULT_EMB_PT))
    parser.add_argument("--idx",  default=str(DEFAULT_IDX_CSV))
    parser.add_argument("--clu",  default=str(DEFAULT_CLU_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    emb_pt  = torch.load(args.emb, map_location="cpu").numpy().astype(np.float32)
    idx_df  = pd.read_csv(args.idx)
    clu_df  = pd.read_csv(args.clu)

    # Merge: index → clusters (on image_path), keep only clustered instances
    df = idx_df.reset_index(drop=True).copy()
    df["row_idx"] = df.index
    df = df.merge(clu_df[["image_path", "cluster_id"]], on="image_path", how="left")
    df = df[df["cluster_id"].notna()].reset_index(drop=True)

    print(f"Loaded {len(emb_pt)} embeddings — using {len(df)} clustered instances "
          f"({len(emb_pt) - len(df)} noise/unclustered dropped)")
    print(f"Clusters: {df['cluster_id'].nunique()}  |  "
          f"Side dist: {df['side_pred'].value_counts().to_dict()}")

    # Encode cluster_id as integer
    cluster_int = pd.factorize(df["cluster_id"])[0]

    results: list[dict] = []
    for side in ("left", "right", "combined"):
        if side == "combined":
            mask = df["side_pred"].isin(["left", "right"])
        else:
            mask = df["side_pred"] == side

        sub = df[mask].copy()
        if len(sub) < 10:
            print(f"\n[{side}] Only {len(sub)} images — skipping (too few for reliable metrics).")
            continue

        sub_emb    = emb_pt[sub["row_idx"].values]
        sub_labels = cluster_int[sub.index.values]

        # L2-normalize (should already be normalized, but re-normalize for safety)
        norms = np.linalg.norm(sub_emb, axis=1, keepdims=True)
        sub_emb = sub_emb / np.clip(norms, 1e-9, None)

        m = compute_metrics(sub_emb, sub_labels, side)
        if m:
            results.append(m)

    if not results:
        print("No results computed — check data.")
        return

    # --- Report ---
    print("\n" + "="*60)
    print("  DINOv2 SIMILARITY HYPOTHESIS VALIDATION")
    print("  Proxy: manually-reviewed cluster assignments (unified_round5)")
    print("="*60)

    verdicts = []
    for m in results:
        v = print_metrics(m)
        if m["side"] != "combined":
            verdicts.append(v)

    # Overall recommendation
    print("\n" + "="*60)
    if all(v == "PASS" for v in verdicts):
        print("  RECOMMENDATION: PASS — proceed with building identification tool.")
    elif any(v == "FAIL" for v in verdicts):
        print("  RECOMMENDATION: FAIL — stop. Show these results to user and")
        print("  escalate to Opus before proceeding with tasks #1-4.")
    else:
        print("  RECOMMENDATION: MARGINAL — results are borderline. Show to user")
        print("  and ask Opus whether to proceed or try a stronger embedding model.")
    print("="*60)

    print_caveats()

    # --- Plots ---
    side_results = [m for m in results if m["side"] != "combined"]
    plot_histograms(side_results, out_dir / "similarity_hypothesis.png")
    plot_roc(side_results,        out_dir / "similarity_roc.png")


if __name__ == "__main__":
    main()
