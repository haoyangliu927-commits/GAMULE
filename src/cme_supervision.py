from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CMEExpression:
    matrix: np.ndarray
    gene_names: list[str]


@dataclass
class CMESupervisionResult:
    negative_mask: np.ndarray
    positive_mask: np.ndarray
    cme_binary: np.ndarray
    jaccard_matrix: np.ndarray
    jaccard_neighbor_mask: np.ndarray
    positive_candidate_mask: np.ndarray
    stats: dict[str, Any]


def load_expression_from_h5ad(
    adata_path: str | Path,
    *,
    layer: str | None = None,
    num_genes: int | None = None,
) -> CMEExpression:
    import scanpy as sc
    from scipy import sparse

    adata = sc.read_h5ad(str(adata_path))
    x = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(x):
        x = x.toarray()

    if num_genes is not None:
        x = x[:, :num_genes]
        gene_names = list(adata.var_names[:num_genes])
    else:
        gene_names = list(adata.var_names)

    return CMEExpression(matrix=np.asarray(x.T, dtype=np.float32), gene_names=gene_names)


def _compute_cme_matrix_numpy(expression_gene_by_cell: np.ndarray) -> np.ndarray:
    x = np.asarray(expression_gene_by_cell, dtype=np.float32)
    gene_sums = x.sum(axis=1).astype(np.float32)
    safe_sums = np.where(gene_sums > 0, gene_sums, 1.0).astype(np.float32)
    cme = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)

    for i in range(x.shape[0]):
        min_sums = np.minimum(x[i], x).sum(axis=1)
        ratio_i = min_sums / safe_sums[i]
        ratio_j = min_sums / safe_sums
        cme[i] = 1.0 - np.maximum(ratio_i, ratio_j)

    cme[gene_sums == 0, :] = 0.0
    cme[:, gene_sums == 0] = 0.0
    return cme


def compute_cme_matrix(
    expression_gene_by_cell: np.ndarray,
    *,
    normalize: bool = False,
    use_numba: bool = True,
) -> np.ndarray:
    x = np.asarray(expression_gene_by_cell, dtype=np.float32)

    if use_numba:
        try:
            from CME_CPU import CME_cpu

            cme = CME_cpu(x, normalize=normalize)
        except Exception:
            cme = _compute_cme_matrix_numpy(x)
    else:
        cme = _compute_cme_matrix_numpy(x)

    cme = np.asarray(cme, dtype=np.float32)
    cme = np.clip((cme + cme.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(cme, 0.0)
    return cme


def shuffle_expression_by_gene(
    expression_gene_by_cell: np.ndarray,
    *,
    rng: np.random.Generator,
    normalize_columns: bool = True,
    target_library_size: float = 1e4,
) -> np.ndarray:
    x = np.asarray(expression_gene_by_cell, dtype=np.float32)
    shuffled = np.empty_like(x)
    for gene_idx in range(x.shape[0]):
        shuffled[gene_idx] = x[gene_idx, rng.permutation(x.shape[1])]

    if normalize_columns:
        col_sums = shuffled.sum(axis=0)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        shuffled = shuffled / col_sums[None, :] * target_library_size
    return shuffled.astype(np.float32)


def compute_cme_pvalues(
    expression_gene_by_cell: np.ndarray,
    cme_matrix: np.ndarray,
    *,
    n_permutations: int = 50,
    seed: int = 0,
    normalize_null_columns: bool = True,
    use_numba: bool = True,
) -> np.ndarray:
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    rng = np.random.default_rng(seed)
    observed = np.asarray(cme_matrix, dtype=np.float32)
    null_ge_observed = np.zeros(observed.shape, dtype=np.int32)

    for _ in range(n_permutations):
        shuffled = shuffle_expression_by_gene(
            expression_gene_by_cell,
            rng=rng,
            normalize_columns=normalize_null_columns,
        )
        null_cme = compute_cme_matrix(shuffled, normalize=False, use_numba=use_numba)
        null_ge_observed += null_cme >= observed

    pvalues = (null_ge_observed + 1.0) / (n_permutations + 1.0)
    np.fill_diagonal(pvalues, 1.0)
    return pvalues.astype(np.float32)


def cme_to_binary_exclusion(
    cme_matrix: np.ndarray,
    pval_matrix: np.ndarray | None,
    *,
    cme_threshold: float,
    pvalue_threshold: float | None,
    pvalue_mode: str = "pvalue",
) -> np.ndarray:
    cme = np.asarray(cme_matrix)
    cme_pass = cme > cme_threshold

    if pval_matrix is None or pvalue_threshold is None:
        mask = cme_pass
    else:
        pvals = np.asarray(pval_matrix)
        if pvalue_mode == "pvalue":
            p_pass = pvals < pvalue_threshold
        elif pvalue_mode == "empirical_cdf":
            p_pass = pvals > (1.0 - pvalue_threshold)
        else:
            raise ValueError("pvalue_mode must be 'pvalue' or 'empirical_cdf'.")
        mask = cme_pass & p_pass

    mask = np.asarray(mask, dtype=bool)
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    return mask


def compute_jaccard_matrix(cme_binary: np.ndarray) -> np.ndarray:
    binary = np.asarray(cme_binary, dtype=bool)
    values = binary.astype(np.float32)
    intersection = values @ values.T
    row_sums = values.sum(axis=1, dtype=np.float32)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )
    np.fill_diagonal(jaccard, 0.0)
    return jaccard.astype(np.float32)


def topk_jaccard_neighbors(
    jaccard_matrix: np.ndarray,
    *,
    jaccard_threshold: float,
    top_k: int,
) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    jaccard = np.asarray(jaccard_matrix, dtype=np.float32)
    neighbor_mask = np.zeros(jaccard.shape, dtype=bool)
    for gene_idx in range(jaccard.shape[0]):
        candidates = np.flatnonzero(jaccard[gene_idx] > jaccard_threshold)
        candidates = candidates[candidates != gene_idx]
        if candidates.size == 0:
            continue
        order = np.argsort(jaccard[gene_idx, candidates], kind="mergesort")[::-1]
        selected = candidates[order[:top_k]]
        neighbor_mask[gene_idx, selected] = True

    np.fill_diagonal(neighbor_mask, False)
    return neighbor_mask


def expand_neighbors_to_second_hop(neighbor_mask: np.ndarray) -> np.ndarray:
    direct = np.asarray(neighbor_mask, dtype=bool)
    second_hop = (direct.astype(np.int16) @ direct.astype(np.int16)) > 0
    candidate = direct | second_hop
    np.fill_diagonal(candidate, False)
    return candidate


def build_supervision_from_cme(
    cme_matrix: np.ndarray,
    pval_matrix: np.ndarray | None = None,
    *,
    cme_threshold: float = 0.9,
    pvalue_threshold: float | None = 0.05,
    jaccard_threshold: float = 0.7,
    top_k: int = 10,
    pvalue_mode: str = "pvalue",
) -> CMESupervisionResult:
    negative_mask = cme_to_binary_exclusion(
        cme_matrix,
        pval_matrix,
        cme_threshold=cme_threshold,
        pvalue_threshold=pvalue_threshold,
        pvalue_mode=pvalue_mode,
    )
    jaccard = compute_jaccard_matrix(negative_mask)
    neighbors = topk_jaccard_neighbors(
        jaccard,
        jaccard_threshold=jaccard_threshold,
        top_k=top_k,
    )
    positive_candidate = expand_neighbors_to_second_hop(neighbors)
    positive_mask = positive_candidate | positive_candidate.T
    positive_mask = positive_mask & (~negative_mask)
    np.fill_diagonal(positive_mask, False)

    upper = np.triu_indices(negative_mask.shape[0], k=1)
    stats = {
        "num_genes": int(negative_mask.shape[0]),
        "cme_threshold": float(cme_threshold),
        "pvalue_threshold": None if pvalue_threshold is None else float(pvalue_threshold),
        "jaccard_threshold": float(jaccard_threshold),
        "top_k": int(top_k),
        "negative_pairs": int(negative_mask[upper].sum()),
        "positive_pairs": int(positive_mask[upper].sum()),
        "positive_negative_overlap": int((positive_mask & negative_mask).sum()),
        "genes_with_negative_signal": int(negative_mask.any(axis=0).sum()),
        "genes_with_positive_signal": int(positive_mask.any(axis=0).sum()),
    }

    return CMESupervisionResult(
        negative_mask=negative_mask,
        positive_mask=positive_mask,
        cme_binary=negative_mask,
        jaccard_matrix=jaccard,
        jaccard_neighbor_mask=neighbors,
        positive_candidate_mask=positive_candidate,
        stats=stats,
    )


def mask_to_pairs(mask: np.ndarray, *, upper_only: bool = True) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if upper_only:
        row_idx, col_idx = np.where(np.triu(mask, k=1))
    else:
        row_idx, col_idx = np.where(mask)
    return np.column_stack([row_idx, col_idx]).astype(np.int64)


def save_supervision_npz(
    path: str | Path,
    *,
    cme_matrix: np.ndarray,
    pval_matrix: np.ndarray | None,
    supervision: CMESupervisionResult,
) -> None:
    save_items = {
        "cme_matrix": np.asarray(cme_matrix, dtype=np.float32),
        "negative_mask": supervision.negative_mask.astype(bool),
        "positive_mask": supervision.positive_mask.astype(bool),
        "jaccard_matrix": supervision.jaccard_matrix.astype(np.float32),
        "jaccard_neighbor_mask": supervision.jaccard_neighbor_mask.astype(bool),
        "positive_pairs": mask_to_pairs(supervision.positive_mask),
        "negative_pairs": mask_to_pairs(supervision.negative_mask),
    }
    if pval_matrix is not None:
        save_items["pval_matrix"] = np.asarray(pval_matrix, dtype=np.float32)
    np.savez_compressed(path, **save_items)
