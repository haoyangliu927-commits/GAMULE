from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cme_supervision import (  # noqa: E402
    build_supervision_from_cme,
    compute_cme_matrix,
    compute_cme_pvalues,
    load_expression_from_h5ad,
    save_supervision_npz,
)
from src.cme_visualization import plot_cme_supervision_heatmaps  # noqa: E402
from src.supervision_pipeline import run_supervised_hyperedges, summarize_unassigned_genes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CME/Jaccard supervision and draw heatmaps.")
    parser.add_argument("--adata", type=Path, default=REPO_ROOT / "datasets/adata_672.h5ad")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--num-genes", type=int, default=420)
    parser.add_argument("--cme-threshold", type=float, default=0.9)
    parser.add_argument("--pvalue-threshold", type=float, default=0.05)
    parser.add_argument("--jaccard-threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expression = load_expression_from_h5ad(args.adata, num_genes=args.num_genes)
    cme_matrix = compute_cme_matrix(expression.matrix, normalize=False, use_numba=True)
    pval_matrix = compute_cme_pvalues(
        expression.matrix,
        cme_matrix,
        n_permutations=args.n_permutations,
        seed=args.seed,
        use_numba=True,
    )
    supervision = build_supervision_from_cme(
        cme_matrix,
        pval_matrix,
        cme_threshold=args.cme_threshold,
        pvalue_threshold=args.pvalue_threshold,
        jaccard_threshold=args.jaccard_threshold,
        top_k=args.top_k,
    )

    heatmap_path = args.output_dir / "cme_supervision_heatmaps.png"
    npz_path = args.output_dir / "cme_supervision_matrices.npz"
    stats_path = args.output_dir / "cme_supervision_stats.json"

    plot_cme_supervision_heatmaps(
        cme_matrix=cme_matrix,
        pval_matrix=pval_matrix,
        supervision=supervision,
        save_path=heatmap_path,
    )
    save_supervision_npz(
        npz_path,
        cme_matrix=cme_matrix,
        pval_matrix=pval_matrix,
        supervision=supervision,
    )

    stats = dict(supervision.stats)
    stats["heatmap_path"] = str(heatmap_path)
    stats["npz_path"] = str(npz_path)

    if args.train_epochs > 0:
        torch.manual_seed(args.seed)
        result = run_supervised_hyperedges(
            adata_path=args.adata,
            pos_mask=torch.from_numpy(supervision.positive_mask),
            neg_mask=torch.from_numpy(supervision.negative_mask),
            num_hyperedges=7,
            use_unassigned_hyperedge=True,
            pos_strength=0.5,
            neg_strength=0.0,
            epochs=args.train_epochs,
            lr=0.016,
            entropy_strength=0.001,
            device=args.device,
            seed=args.seed,
        )
        stats["module_summary"] = summarize_unassigned_genes(result)

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
