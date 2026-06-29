from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class CellTypeModuleEvaluation:
    module_cell_type_table: pd.DataFrame
    cell_assignment_table: pd.DataFrame
    cell_type_module_mean_scores: pd.DataFrame
    summary: dict


def _mean_expression_by_gene_set(x, gene_indices: np.ndarray) -> np.ndarray:
    if gene_indices.size == 0:
        return np.zeros(x.shape[0], dtype=np.float64)
    values = x[:, gene_indices]
    means = values.mean(axis=1)
    if sparse.issparse(means):
        means = means.A1
    else:
        means = np.asarray(means).reshape(-1)
    return means.astype(np.float64)


def _normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    values = scores.to_numpy(dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    z = (values - mean) / (std + 1e-8)
    return pd.DataFrame(z, index=scores.index, columns=scores.columns)


def _entropy_from_counts(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    p = values / total
    return float(-(p * np.log(p + 1e-8)).sum() / np.log(len(p) + 1e-8))


def evaluate_cell_type_modules(
    adata,
    gene_assignment,
    *,
    cell_type_column: str,
    garbage_hyperedge_index: int | None = None,
    result_dir: str | Path | None = None,
    exclude_garbage: bool = True,
) -> CellTypeModuleEvaluation:
    """Evaluate learned gene modules against cell-type labels using module activity.

    This does not use tree/XML information. For each hyperedge, cell-level module
    activity is the mean expression of genes assigned to that hyperedge.
    """
    if cell_type_column not in adata.obs:
        raise ValueError(f"{cell_type_column!r} is not present in adata.obs.")

    if isinstance(gene_assignment, pd.DataFrame):
        assignment_df = gene_assignment.copy()
    else:
        assignment_df = pd.DataFrame(
            {
                "gene_index": np.arange(len(gene_assignment), dtype=int),
                "assigned_hyperedge": np.asarray(gene_assignment, dtype=int),
            }
        )

    if "assigned_hyperedge" not in assignment_df:
        raise ValueError("gene_assignment must contain an 'assigned_hyperedge' column.")

    if "gene_index" not in assignment_df:
        if "gene_name" in assignment_df:
            name_to_index = {str(name): idx for idx, name in enumerate(adata.var_names.astype(str))}
            assignment_df["gene_index"] = assignment_df["gene_name"].astype(str).map(name_to_index)
            assignment_df = assignment_df.dropna(subset=["gene_index"]).copy()
            assignment_df["gene_index"] = assignment_df["gene_index"].astype(int)
        else:
            assignment_df["gene_index"] = np.arange(len(assignment_df), dtype=int)

    assignment_df = assignment_df[
        (assignment_df["gene_index"] >= 0) & (assignment_df["gene_index"] < adata.n_vars)
    ].copy()
    assignment_df["assigned_hyperedge"] = assignment_df["assigned_hyperedge"].astype(int)

    cell_types = adata.obs[cell_type_column].astype(str).reset_index(drop=True)
    hyperedges = sorted(assignment_df["assigned_hyperedge"].unique().tolist())
    scored_hyperedges = [
        h
        for h in hyperedges
        if not (exclude_garbage and garbage_hyperedge_index is not None and h == garbage_hyperedge_index)
    ]
    if not scored_hyperedges:
        raise ValueError("No non-garbage hyperedges are available for cell-type evaluation.")

    score_columns = []
    score_values = []
    module_sizes = {}
    for hyperedge in scored_hyperedges:
        gene_indices = assignment_df.loc[
            assignment_df["assigned_hyperedge"] == hyperedge,
            "gene_index",
        ].to_numpy(dtype=int)
        module_sizes[int(hyperedge)] = int(gene_indices.size)
        score_columns.append(f"H{int(hyperedge)}")
        score_values.append(_mean_expression_by_gene_set(adata.X, gene_indices))

    score_matrix = np.vstack(score_values).T
    score_df = pd.DataFrame(score_matrix, columns=score_columns)
    score_z = _normalize_scores(score_df)
    score_z["cell_type"] = cell_types.values

    mean_scores = score_z.groupby("cell_type", observed=False).mean().T
    mean_scores.index.name = "hyperedge"

    module_rows = []
    module_to_cell_type = {}
    for hyperedge, column in zip(scored_hyperedges, score_columns):
        means = mean_scores.loc[column].sort_values(ascending=False)
        dominant = str(means.index[0])
        second = str(means.index[1]) if len(means) > 1 else None
        dominant_mean = float(means.iloc[0])
        second_mean = float(means.iloc[1]) if len(means) > 1 else np.nan
        assigned_mask = score_z[score_columns].idxmax(axis=1).eq(column)
        assigned_types = cell_types[assigned_mask]
        assigned_counts = assigned_types.value_counts()
        assigned_dominant = str(assigned_counts.index[0]) if len(assigned_counts) else None
        assigned_purity = (
            float(assigned_counts.iloc[0] / assigned_counts.sum()) if assigned_counts.sum() else np.nan
        )
        module_to_cell_type[column] = dominant
        module_rows.append(
            {
                "hyperedge": int(hyperedge),
                "score_column": column,
                "num_genes": module_sizes[int(hyperedge)],
                "dominant_cell_type_by_mean_score": dominant,
                "second_cell_type_by_mean_score": second,
                "dominant_mean_zscore": dominant_mean,
                "second_mean_zscore": second_mean,
                "mean_score_gap": dominant_mean - second_mean if second is not None else np.nan,
                "assigned_cells_by_argmax": int(assigned_mask.sum()),
                "assigned_dominant_cell_type": assigned_dominant,
                "assigned_cell_type_purity": assigned_purity,
                "assigned_cell_type_entropy": _entropy_from_counts(assigned_counts),
            }
        )

    best_module = score_z[score_columns].idxmax(axis=1)
    predicted_cell_type = best_module.map(module_to_cell_type).astype(str)
    correct = predicted_cell_type.to_numpy() == cell_types.to_numpy()
    cell_assignment = pd.DataFrame(
        {
            "cell_index": np.arange(adata.n_obs, dtype=int),
            "true_cell_type": cell_types.values,
            "best_hyperedge": best_module.str.replace("H", "", regex=False).astype(int).values,
            "predicted_cell_type": predicted_cell_type.values,
            "best_module_zscore": score_z[score_columns].max(axis=1).values,
            "correct": correct,
        }
    )

    module_table = pd.DataFrame(module_rows)
    accuracy = float(correct.mean()) if correct.size else np.nan
    summary = {
        "obs_column": cell_type_column,
        "num_cells": int(adata.n_obs),
        "num_scored_hyperedges": int(len(scored_hyperedges)),
        "garbage_hyperedge_index": None if garbage_hyperedge_index is None else int(garbage_hyperedge_index),
        "exclude_garbage": bool(exclude_garbage),
        "module_argmax_cell_type_accuracy": accuracy,
        "module_argmax_cell_type_accuracy_percent": f"{accuracy:.2%}" if not np.isnan(accuracy) else None,
        "mean_assigned_cell_type_purity": float(module_table["assigned_cell_type_purity"].mean()),
    }

    if result_dir is not None:
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        module_table.to_csv(result_path / "module_cell_type_table.csv", index=False)
        cell_assignment.to_csv(result_path / "cell_assignment_from_modules.csv", index=False)
        mean_scores.to_csv(result_path / "cell_type_module_mean_scores.csv")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig_width = max(6.0, 0.45 * mean_scores.shape[1] + 2.0)
            fig_height = max(4.0, 0.28 * mean_scores.shape[0] + 1.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                mean_scores,
                cmap="vlag",
                center=0.0,
                linewidths=0.2,
                linecolor="white",
                cbar_kws={"label": "mean module z-score"},
                ax=ax,
            )
            ax.set_xlabel("Cell type")
            ax.set_ylabel("Hyperedge")
            ax.set_title("Module activity by cell type")
            fig.tight_layout()
            fig.savefig(result_path / "cell_type_module_mean_scores_heatmap.png", dpi=240)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - plotting is diagnostic only.
            print(f"Warning: could not save cell-type module heatmap: {exc}")

    return CellTypeModuleEvaluation(
        module_cell_type_table=module_table,
        cell_assignment_table=cell_assignment,
        cell_type_module_mean_scores=mean_scores,
        summary=summary,
    )
