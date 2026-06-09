from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


TARGET_GROUPS = {
    "expr_block_1": ["Pax6", "Cpe", "Gars", "Hpca", "Celf3", "Rgs17", "Tuba1a"],
    "expr_block_2": ["Ndc80", "Racgap1", "Aurkb", "Cdk1", "Prc1", "Cdc20", "Ccnb1"],
    "expr_block_3": ["Dbi", "Mgst1", "Sparc", "Spp1", "Id1", "Vim", "Acot1"],
}


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_representative_sets(text: str) -> list[list[int] | None]:
    if not text.strip():
        return [None]
    sets: list[list[int] | None] = []
    for item in text.split(";"):
        item = item.strip()
        if not item or item.lower() in {"none", "auto"}:
            sets.append(None)
        else:
            sets.append(parse_int_list(item))
    return sets


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Grid search ngn3 GAMULE parameters against heatmap blocks.")
    parser.add_argument("--adata-path", type=Path, default=repo_root / "datasets" / "adata_ngn3_ss.h5ad")
    parser.add_argument("--result-dir", type=Path, default=repo_root / "results" / "ngn3_grid_search")
    parser.add_argument("--cme-thresholds", default="0.55,0.66,0.75")
    parser.add_argument("--jaccard-thresholds", default="0.50,0.65,0.73")
    parser.add_argument("--inclusion-thresholds", default="0.10,0.15")
    parser.add_argument("--hierarchy-strengths", default="0,0.001")
    parser.add_argument("--pos-strengths", default="0.5")
    parser.add_argument("--partial-pos-strengths", default="0.25")
    parser.add_argument("--entropy-strengths", default="0.001")
    parser.add_argument("--representative-sets", default="")
    parser.add_argument("--pvalue-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=297)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--num-gene-modules", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.016)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run the full grid.")
    args = parser.parse_args()
    args.repo_root = repo_root
    return args


def target_labels_for_genes(gene_names: list[str]) -> list[int]:
    lookup = {
        gene: group_idx
        for group_idx, genes in enumerate(TARGET_GROUPS.values())
        for gene in genes
    }
    missing = [gene for gene in gene_names if gene not in lookup]
    if missing:
        raise ValueError(f"Target groups do not cover genes: {missing}")
    return [lookup[gene] for gene in gene_names]


def purity_score(true_labels: list[int], pred_labels: list[int]) -> float:
    total = len(true_labels)
    correct = 0
    for pred in sorted(set(pred_labels)):
        members = [idx for idx, label in enumerate(pred_labels) if label == pred]
        counts: dict[int, int] = {}
        for idx in members:
            counts[true_labels[idx]] = counts.get(true_labels[idx], 0) + 1
        correct += max(counts.values()) if counts else 0
    return correct / total if total else 0.0


def module_gene_table(gene_names: list[str], pred_labels: list[int]) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for gene, label in zip(gene_names, pred_labels):
        table.setdefault(f"H{int(label)}", []).append(gene)
    return dict(sorted(table.items(), key=lambda item: item[0]))


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root))

    from src.cme_supervision import (
        build_supervision_from_cme,
        compute_cme_matrix,
        compute_cme_pvalues,
        load_expression_from_h5ad,
    )
    from src.supervision_pipeline import run_supervised_hyperedges

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(args.adata_path))
    gene_names = list(map(str, adata.var_names))
    true_labels = target_labels_for_genes(gene_names)
    seeds = parse_int_list(args.seeds)
    pvalue_seed = seeds[0] if seeds else 0
    num_genes = len(gene_names)
    num_gene_modules = max(1, min(int(args.num_gene_modules), num_genes))
    num_hyperedges = num_gene_modules + 1

    print(f"[grid] genes={num_genes}, num_gene_modules={num_gene_modules}", flush=True)
    expression = load_expression_from_h5ad(args.adata_path, num_genes=num_genes)
    cme_matrix = compute_cme_matrix(expression.matrix, normalize=False, use_numba=True)
    pval_matrix = compute_cme_pvalues(
        expression.matrix,
        cme_matrix,
        n_permutations=int(args.n_permutations),
        seed=pvalue_seed,
        use_numba=True,
    )

    grid = list(
        itertools.product(
            parse_float_list(args.cme_thresholds),
            parse_float_list(args.jaccard_thresholds),
            parse_float_list(args.inclusion_thresholds),
            parse_float_list(args.hierarchy_strengths),
            parse_float_list(args.pos_strengths),
            parse_float_list(args.partial_pos_strengths),
            parse_float_list(args.entropy_strengths),
            seeds,
            parse_representative_sets(args.representative_sets),
        )
    )
    if args.max_runs and args.max_runs > 0:
        grid = grid[: int(args.max_runs)]

    rows = []
    for run_idx, values in enumerate(grid, start=1):
        (
            t_cme,
            t_jaccard,
            t_inclusion,
            hierarchy_strength,
            pos_strength,
            partial_pos_strength,
            entropy_strength,
            seed,
            representative_indices,
        ) = values
        print(
            f"[grid] {run_idx}/{len(grid)} "
            f"t_CME={t_cme}, t_Jaccard={t_jaccard}, "
            f"t_inclusion={t_inclusion}, hierarchy_strength={hierarchy_strength}, "
            f"pos_strength={pos_strength}, partial_pos_strength={partial_pos_strength}, "
            f"entropy_strength={entropy_strength}, seed={seed}, "
            f"representatives={representative_indices}",
            flush=True,
        )
        try:
            supervision = build_supervision_from_cme(
                cme_matrix,
                pval_matrix,
                cme_threshold=t_cme,
                pvalue_threshold=float(args.pvalue_threshold),
                jaccard_threshold=t_jaccard,
                top_k=int(args.top_k),
                inclusion_threshold=t_inclusion,
            )
            pos_mask = torch.from_numpy(supervision.positive_mask)
            neg_mask = torch.from_numpy(supervision.negative_mask)
            partial_pos_mask = torch.from_numpy(supervision.inclusion_partial_mask)
            directed_inclusion_mask = torch.from_numpy(supervision.inclusion_directed_mask)

            result = run_supervised_hyperedges(
                adata=adata,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
                partial_pos_mask=partial_pos_mask,
                directed_inclusion_mask=directed_inclusion_mask,
                num_genes=num_genes,
                num_hyperedges=num_hyperedges,
                use_unassigned_hyperedge=True,
                pos_strength=pos_strength,
                partial_pos_strength=partial_pos_strength,
                neg_strength=0.0,
                hierarchy_strength=hierarchy_strength,
                hierarchy_margin=0.0,
                hierarchy_min_direction_weight=0.0,
                epochs=int(args.epochs),
                lr=float(args.lr),
                entropy_strength=entropy_strength,
                ranges_map=None,
                representative_indices=representative_indices,
                device="auto",
                seed=seed,
            )

            pred_labels = result.partition[:, : result.num_gene_modules].argmax(dim=1).cpu().tolist()
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            purity = purity_score(true_labels, pred_labels)
            modules = module_gene_table(gene_names, pred_labels)
            row = {
                "run_idx": run_idx,
                "status": "ok",
                "t_CME": t_cme,
                "t_Jaccard": t_jaccard,
                "t_inclusion": t_inclusion,
                "hierarchy_strength": hierarchy_strength,
                "pos_strength": pos_strength,
                "partial_pos_strength": partial_pos_strength,
                "entropy_strength": entropy_strength,
                "seed": seed,
                "representative_indices": (
                    "auto" if representative_indices is None else ",".join(map(str, representative_indices))
                ),
                "ari": float(ari),
                "nmi": float(nmi),
                "purity": float(purity),
                "final_loss": float(result.losses[-1]),
                "negative_pairs": supervision.stats["negative_pairs"],
                "positive_pairs": supervision.stats["positive_pairs"],
                "inclusion_directed_pairs": supervision.stats["inclusion_directed_pairs"],
                "inclusion_partial_pairs": supervision.stats["inclusion_partial_pairs"],
                "modules_json": json.dumps(modules, ensure_ascii=False),
            }
        except Exception as exc:
            row = {
                "run_idx": run_idx,
                "status": "error",
                "t_CME": t_cme,
                "t_Jaccard": t_jaccard,
                "t_inclusion": t_inclusion,
                "hierarchy_strength": hierarchy_strength,
                "pos_strength": pos_strength,
                "partial_pos_strength": partial_pos_strength,
                "entropy_strength": entropy_strength,
                "seed": seed,
                "representative_indices": (
                    "auto" if representative_indices is None else ",".join(map(str, representative_indices))
                ),
                "error": repr(exc),
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_dir / "grid_results_partial.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    if "ari" in df.columns:
        df = df.sort_values(["ari", "nmi", "purity"], ascending=False, na_position="last")
    df.to_csv(result_dir / "grid_results.csv", index=False, encoding="utf-8-sig")
    summary = {
        "target_groups": TARGET_GROUPS,
        "config": {
            "epochs": int(args.epochs),
            "num_gene_modules": int(args.num_gene_modules),
            "n_permutations": int(args.n_permutations),
            "grid_size": len(grid),
        },
        "top_results": df.head(10).to_dict(orient="records"),
        "outputs": {
            "result_dir": str(result_dir),
            "grid_results_csv": str(result_dir / "grid_results.csv"),
            "partial_csv": str(result_dir / "grid_results_partial.csv"),
        },
    }
    with open(result_dir / "grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print("JSON_RESULT_START", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print("JSON_RESULT_END", flush=True)
    print("[grid] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
