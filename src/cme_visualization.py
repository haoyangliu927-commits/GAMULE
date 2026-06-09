from __future__ import annotations

from pathlib import Path

import numpy as np

from src.cme_supervision import CMESupervisionResult


def _as_float_panel(matrix):
    return np.asarray(matrix, dtype=np.float32)


def plot_cme_supervision_heatmaps(
    *,
    cme_matrix: np.ndarray,
    pval_matrix: np.ndarray | None,
    supervision: CMESupervisionResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    panels = [
        (_as_float_panel(cme_matrix), "CME matrix", "viridis", None),
        (
            _as_float_panel(pval_matrix) if pval_matrix is not None else None,
            "CME p-value",
            "magma_r",
            None,
        ),
        (supervision.cme_binary.astype(float), "CME true / negative", "Blues", None),
        (_as_float_panel(supervision.jaccard_matrix), "Jaccard matrix", "viridis", None),
        (supervision.jaccard_neighbor_mask.astype(float), "Top-k Jaccard neighbors", "Greens", None),
        (supervision.positive_mask.astype(float), "Positive supervision", "Reds", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, (matrix, title, cmap, limits) in zip(axes.ravel(), panels):
        if matrix is None:
            ax.axis("off")
            ax.set_title(title + " (not provided)")
            continue
        image = ax.imshow(matrix, cmap=cmap, interpolation="nearest", aspect="auto")
        if limits is not None:
            image.set_clim(*limits)
        ax.set_title(title)
        ax.set_xlabel("Gene index")
        ax.set_ylabel("Gene index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "CME -> Jaccard -> supervision masks\n"
        f"pos={supervision.stats['positive_pairs']}, neg={supervision.stats['negative_pairs']}",
        fontsize=14,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_combined_cme_supervision_heatmaps(
    *,
    cme_matrix: np.ndarray,
    pval_matrix: np.ndarray | None,
    supervision: CMESupervisionResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    panels = [
        (_as_float_panel(cme_matrix), "CME matrix", "viridis", None),
        (
            _as_float_panel(pval_matrix) if pval_matrix is not None else None,
            "CME p-value",
            "magma_r",
            None,
        ),
        (supervision.negative_mask.astype(float), "Negative supervision", "Blues", (0.0, 1.0)),
        (_as_float_panel(supervision.jaccard_matrix), "Jaccard matrix", "viridis", None),
        (
            supervision.jaccard_neighbor_mask.astype(float),
            "Top-k Jaccard neighbors",
            "Greens",
            (0.0, 1.0),
        ),
        (supervision.positive_mask.astype(float), "Positive supervision", "Reds", (0.0, 1.0)),
        (
            None
            if supervision.inclusion_score_matrix is None
            else _as_float_panel(supervision.inclusion_score_matrix),
            "Inclusion score matrix",
            "viridis",
            None,
        ),
        (
            None
            if supervision.inclusion_directed_mask is None
            else supervision.inclusion_directed_mask.astype(float),
            "Directed inclusion",
            "Purples",
            (0.0, 1.0),
        ),
        (
            None
            if supervision.inclusion_partial_mask is None
            else supervision.inclusion_partial_mask.astype(float),
            "Hierarchy positive supervision",
            "Oranges",
            (0.0, 1.0),
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    for ax, (matrix, title, cmap, limits) in zip(axes.ravel(), panels):
        if matrix is None:
            ax.axis("off")
            ax.set_title(title + " (not available)")
            continue
        image = ax.imshow(matrix, cmap=cmap, interpolation="nearest", aspect="auto")
        if limits is not None:
            image.set_clim(*limits)
        ax.set_title(title)
        ax.set_xlabel("Gene index")
        ax.set_ylabel("Gene index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    stats = supervision.stats
    fig.suptitle(
        "CME negative, Jaccard positive, and hierarchy positive supervision\n"
        f"pos={stats.get('positive_pairs', 0)}, "
        f"neg={stats.get('negative_pairs', 0)}, "
        f"inclusion={stats.get('inclusion_partial_pairs', 0)}",
        fontsize=14,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
