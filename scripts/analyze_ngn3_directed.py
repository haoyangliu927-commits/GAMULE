from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import scanpy as sc
import torch


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze ngn3 h5ad with three-way CME supervision and directed hierarchy regularization.",
    )
    parser.add_argument("--adata-path", type=Path, default=repo_root / "datasets" / "adata_ngn3_ss.h5ad")
    parser.add_argument("--result-dir", type=Path, default=repo_root / "results" / "ngn3_directed_hierarchy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cme-threshold", type=float, default=0.66)
    parser.add_argument("--pvalue-threshold", type=float, default=0.05)
    parser.add_argument("--jaccard-threshold", type=float, default=0.73)
    parser.add_argument("--top-k", type=int, default=297)
    parser.add_argument("--inclusion-threshold", type=float, default=0.15)
    parser.add_argument("--hierarchy-strength", type=float, default=0.001)
    parser.add_argument("--hierarchy-margin", type=float, default=0.0)
    parser.add_argument("--hierarchy-min-direction-weight", type=float, default=0.0)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--num-gene-modules", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.016)
    parser.add_argument("--entropy-strength", type=float, default=0.001)
    args = parser.parse_args()
    args.repo_root = repo_root
    return args


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root))

    from src.cme_supervision import (
        build_supervision_from_cme,
        compute_cme_matrix,
        compute_cme_pvalues,
        load_expression_from_h5ad,
        save_supervision_npz,
    )
    from src.loss import directed_hyperedge_inclusion_loss
    from src.metagene_tree import (
        build_metagene_tree_from_result,
        plot_metagene_tree,
        score_cell_types_from_metagene_tree,
        validate_metagene_tree_result,
    )
    from src.supervision_pipeline import run_supervised_hyperedges, summarize_unassigned_genes

    adata_path = Path(args.adata_path)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ngn3] reading {adata_path}", flush=True)
    adata = sc.read_h5ad(str(adata_path))
    num_cells = int(adata.n_obs)
    num_genes = int(adata.n_vars)
    num_gene_modules = max(1, min(int(args.num_gene_modules), num_genes))
    num_hyperedges = num_gene_modules + 1

    print(f"[ngn3] shape={num_cells} cells x {num_genes} genes", flush=True)
    print(f"[ngn3] obs columns={list(map(str, adata.obs.columns))}", flush=True)
    print(f"[ngn3] genes={list(map(str, adata.var_names))}", flush=True)

    expression = load_expression_from_h5ad(adata_path, num_genes=num_genes)
    print("[ngn3] computing CME and p-values", flush=True)
    cme_matrix = compute_cme_matrix(expression.matrix, normalize=False, use_numba=True)
    pval_matrix = compute_cme_pvalues(
        expression.matrix,
        cme_matrix,
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
        use_numba=True,
    )

    print("[ngn3] building three-way supervision", flush=True)
    supervision = build_supervision_from_cme(
        cme_matrix,
        pval_matrix,
        cme_threshold=float(args.cme_threshold),
        pvalue_threshold=float(args.pvalue_threshold),
        jaccard_threshold=float(args.jaccard_threshold),
        top_k=int(args.top_k),
        inclusion_threshold=float(args.inclusion_threshold),
    )
    save_supervision_npz(
        result_dir / "ngn3_three_way_supervision_matrices.npz",
        cme_matrix=cme_matrix,
        pval_matrix=pval_matrix,
        supervision=supervision,
    )

    pos_mask = torch.from_numpy(supervision.positive_mask)
    neg_mask = torch.from_numpy(supervision.negative_mask)
    partial_pos_mask = torch.from_numpy(supervision.inclusion_partial_mask)
    directed_inclusion_mask = torch.from_numpy(supervision.inclusion_directed_mask)

    print("[ngn3] training supervised hyperedges", flush=True)
    result = run_supervised_hyperedges(
        adata=adata,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        partial_pos_mask=partial_pos_mask,
        directed_inclusion_mask=directed_inclusion_mask,
        num_genes=num_genes,
        num_hyperedges=num_hyperedges,
        use_unassigned_hyperedge=True,
        pos_strength=0.5,
        partial_pos_strength=0.25,
        neg_strength=0.0,
        hierarchy_strength=float(args.hierarchy_strength),
        hierarchy_margin=float(args.hierarchy_margin),
        hierarchy_min_direction_weight=float(args.hierarchy_min_direction_weight),
        epochs=int(args.epochs),
        lr=float(args.lr),
        entropy_strength=float(args.entropy_strength),
        ranges_map=None,
        device="auto",
        seed=int(args.seed),
    )

    meaningful_partition = result.partition[:, : result.num_gene_modules]
    final_hierarchy_loss = directed_hyperedge_inclusion_loss(
        meaningful_partition,
        neg_mask.to(meaningful_partition.device),
        directed_inclusion_mask.to(meaningful_partition.device),
    ).item()

    print("[ngn3] building metagene tree", flush=True)
    reference_adata = adata if "gene_module" in adata.var.columns else None
    metagene_tree = build_metagene_tree_from_result(
        result,
        expression.matrix,
        adata=reference_adata,
        gene_names=expression.gene_names,
        reference_column="gene_module",
        cme_threshold=float(args.cme_threshold),
        assignment="argmax",
        aggregation="sum",
        child_strategy="max_degree_ties",
    )
    tree_validation = validate_metagene_tree_result(metagene_tree)
    plot_metagene_tree(
        metagene_tree,
        save_path=result_dir / "ngn3_metagene_cme_tree.png",
    )

    labels = metagene_tree.node_labels
    module_rows = []
    for node_idx, module_idx in enumerate(metagene_tree.module_ids):
        gene_names = metagene_tree.module_gene_names.get(module_idx, [])
        module_rows.append(
            {
                "node_index": int(node_idx),
                "hyperedge": int(module_idx),
                "num_genes": len(gene_names),
                "genes": ";".join(map(str, gene_names)),
                "label": labels[node_idx],
            }
        )
    pd.DataFrame(module_rows).to_csv(
        result_dir / "ngn3_gene_modules.csv",
        index=False,
        encoding="utf-8-sig",
    )

    edge_rows = [
        {
            "parent": labels[parent],
            "child": labels[child],
            "parent_idx": int(parent),
            "child_idx": int(child),
        }
        for parent, child in metagene_tree.tree_edges
    ]
    pd.DataFrame(edge_rows).to_csv(
        result_dir / "ngn3_tree_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        metagene_tree.metagene_cme,
        index=labels[:-1],
        columns=labels[:-1],
    ).to_csv(result_dir / "ngn3_metagene_cme.csv", encoding="utf-8-sig")

    cell_type_summary = None
    if "clusters" in adata.obs:
        cell_type_result = score_cell_types_from_metagene_tree(
            metagene_tree,
            adata=adata,
            obs_column="clusters",
        )
        cell_type_summary = cell_type_result.stats
        pd.DataFrame(cell_type_result.metagene_cell_type_table).to_csv(
            result_dir / "ngn3_metagene_cell_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "adata_path": str(adata_path),
        "shape": [num_cells, num_genes],
        "obs_columns": list(map(str, adata.obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "var_names": list(map(str, adata.var_names)),
        "config": {
            "seed": int(args.seed),
            "cme_threshold": float(args.cme_threshold),
            "pvalue_threshold": float(args.pvalue_threshold),
            "jaccard_threshold": float(args.jaccard_threshold),
            "top_k": int(args.top_k),
            "inclusion_threshold": float(args.inclusion_threshold),
            "hierarchy_strength": float(args.hierarchy_strength),
            "hierarchy_margin": float(args.hierarchy_margin),
            "hierarchy_min_direction_weight": float(args.hierarchy_min_direction_weight),
            "n_permutations": int(args.n_permutations),
            "num_gene_modules": int(num_gene_modules),
            "epochs": int(args.epochs),
        },
        "supervision_stats": supervision.stats,
        "mask_entries": {
            "full_pos": int(pos_mask.sum().item()),
            "partial_pos": int(partial_pos_mask.sum().item()),
            "directed_inclusion": int(directed_inclusion_mask.sum().item()),
            "neg": int(neg_mask.sum().item()),
        },
        "supervision_mode": result.supervision_mode,
        "relation_targets": result.relation_targets,
        "final_total_loss": float(result.losses[-1]),
        "final_hierarchy_directed_loss": float(final_hierarchy_loss),
        "unassigned": summarize_unassigned_genes(result),
        "metagene_tree_stats": metagene_tree.stats,
        "tree_validation": tree_validation,
        "tree_edges": edge_rows,
        "modules": module_rows,
        "cell_type_summary": cell_type_summary,
        "outputs": {
            "result_dir": str(result_dir),
            "summary_json": str(result_dir / "ngn3_analysis_summary.json"),
            "modules_csv": str(result_dir / "ngn3_gene_modules.csv"),
            "tree_edges_csv": str(result_dir / "ngn3_tree_edges.csv"),
            "metagene_cme_csv": str(result_dir / "ngn3_metagene_cme.csv"),
            "metagene_cell_scores_csv": str(result_dir / "ngn3_metagene_cell_scores.csv"),
            "tree_png": str(result_dir / "ngn3_metagene_cme_tree.png"),
        },
    }
    with open(result_dir / "ngn3_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("JSON_RESULT_START", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print("JSON_RESULT_END", flush=True)
    print("[ngn3] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
