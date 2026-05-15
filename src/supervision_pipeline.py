from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.loss import resolve_relation_supervision
from src.train import train_embedding_weakpos0


DEFAULT_HYPEREDGE_RANGES: dict[int, list[tuple[int, int]]] = {
    7: [(0, 100), (100, 200), (200, 230), (230, 260), (260, 360), (360, 390), (390, 420)],
    6: [(0, 100), (100, 200), (200, 260), (260, 360), (360, 390), (390, 420)],
}


@dataclass
class PCAInitialization:
    init_gene_emb: torch.Tensor
    init_hyperedge_emb: torch.Tensor
    init_partition: torch.Tensor
    representative_indices: list[int]


@dataclass
class SupervisedHypergraphResult:
    partition: torch.Tensor
    hyperedge_emb: torch.Tensor
    gene_emb: torch.Tensor
    losses: list[float]
    init_gene_emb: torch.Tensor
    init_hyperedge_emb: torch.Tensor
    init_partition: torch.Tensor
    representative_indices: list[int]
    supervision_mode: str
    relation_targets: dict[str, float]
    num_hyperedges: int
    num_gene_modules: int
    null_hyperedge_index: int | None
    supervised_gene_mask: torch.Tensor
    unsupervised_gene_mask: torch.Tensor
    device: str


def centers_from_ranges(ranges: Sequence[tuple[int, int]]) -> list[int]:
    return [(left + right) // 2 for left, right in ranges]


def evenly_spaced_representatives(num_genes: int, num_hyperedges: int) -> list[int]:
    if num_hyperedges <= 0:
        raise ValueError("num_hyperedges must be positive.")
    if num_hyperedges > num_genes:
        raise ValueError("num_hyperedges cannot be larger than num_genes.")
    return torch.linspace(0, num_genes - 1, steps=num_hyperedges).round().long().tolist()


def choose_representative_indices(
    *,
    num_genes: int,
    num_hyperedges: int,
    ranges_map: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    representative_indices: Sequence[int] | None = None,
) -> list[int]:
    if representative_indices is not None:
        indices = [int(idx) for idx in representative_indices]
    elif ranges_map is not None and num_hyperedges in ranges_map:
        indices = centers_from_ranges(ranges_map[num_hyperedges])
    else:
        indices = evenly_spaced_representatives(num_genes, num_hyperedges)

    if len(indices) != num_hyperedges:
        raise ValueError(
            f"Expected {num_hyperedges} representative indices, got {len(indices)}."
        )
    if min(indices) < 0 or max(indices) >= num_genes:
        raise ValueError("representative_indices contains an index outside num_genes.")
    return indices


def _as_bool_mask(mask, *, name: str) -> torch.Tensor:
    if mask is None:
        raise ValueError(f"{name} is required.")
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape {tuple(mask.shape)}.")
    return mask.bool()


def _infer_num_genes(*masks: torch.Tensor | None) -> int:
    shapes = [tuple(mask.shape) for mask in masks if mask is not None]
    if not shapes:
        raise ValueError("At least one supervision mask is required.")
    if len(set(shapes)) != 1:
        raise ValueError(f"All supervision masks must have the same shape, got {shapes}.")
    return int(shapes[0][0])


def dropout_bool_mask(mask: torch.Tensor, dropout_rate: float = 0.0) -> torch.Tensor:
    if dropout_rate <= 0:
        return mask
    if dropout_rate >= 1:
        return torch.zeros_like(mask, dtype=torch.bool)
    keep_mask = torch.rand(mask.shape, device=mask.device) > dropout_rate
    return mask.bool() & keep_mask


def supervised_gene_mask_from_relations(
    relation_masks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    first_mask = next(iter(relation_masks.values()))
    covered = torch.zeros(first_mask.shape[0], dtype=torch.bool, device=first_mask.device)
    for mask in relation_masks.values():
        mask = mask.bool()
        covered = covered | mask.any(dim=0) | mask.any(dim=1)
    return covered


def append_unassigned_hyperedge(
    *,
    partition_meaningful: torch.Tensor,
    hyperedge_emb_meaningful: torch.Tensor,
    supervised_gene_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    meaningful_count = partition_meaningful.shape[1]
    null_index = meaningful_count

    supervised_gene_mask = supervised_gene_mask.to(partition_meaningful.device).bool()
    full_partition = torch.zeros(
        partition_meaningful.shape[0],
        meaningful_count + 1,
        device=partition_meaningful.device,
        dtype=partition_meaningful.dtype,
    )
    full_partition[supervised_gene_mask, :meaningful_count] = partition_meaningful[
        supervised_gene_mask
    ]
    full_partition[~supervised_gene_mask, null_index] = 1.0

    null_hyperedge = torch.zeros(
        1,
        hyperedge_emb_meaningful.shape[1],
        device=hyperedge_emb_meaningful.device,
        dtype=hyperedge_emb_meaningful.dtype,
    )
    full_hyperedge_emb = torch.cat([hyperedge_emb_meaningful, null_hyperedge], dim=0)

    # The null edge is intentionally excluded from this multiplication.
    # Unsupervised rows become zero vectors, so the next similarity multiplication
    # cannot leak the null edge into meaningful gene programs.
    hyperedge_emb_norm = F.normalize(hyperedge_emb_meaningful, p=2, dim=1)
    full_gene_emb = full_partition[:, :meaningful_count] @ hyperedge_emb_norm

    return full_partition, full_hyperedge_emb, full_gene_emb, null_index


def build_pca_initialization(
    *,
    adata_path: str | Path | None = None,
    adata=None,
    num_genes: int,
    embed_dim: int = 16,
    num_hyperedges: int = 6,
    ranges_map: Mapping[int, Sequence[tuple[int, int]]] | None = DEFAULT_HYPEREDGE_RANGES,
    representative_indices: Sequence[int] | None = None,
) -> PCAInitialization:
    if adata is None:
        if adata_path is None:
            raise ValueError("Either adata or adata_path must be provided.")
        import scanpy as sc

        adata = sc.read_h5ad(str(adata_path))

    pcs = adata.varm.get("PCs", None)
    if pcs is None or pcs.shape[1] < embed_dim:
        import scanpy as sc

        sc.pp.pca(adata, n_comps=embed_dim)
        pcs = adata.varm["PCs"]

    if pcs.shape[0] < num_genes:
        raise ValueError(f"adata has only {pcs.shape[0]} genes, but num_genes={num_genes}.")

    reps = choose_representative_indices(
        num_genes=num_genes,
        num_hyperedges=num_hyperedges,
        ranges_map=ranges_map,
        representative_indices=representative_indices,
    )

    init_gene_emb = torch.from_numpy(pcs[:num_genes, :embed_dim]).float()
    init_hyperedge_emb = init_gene_emb[reps]

    gene_emb_norm = F.normalize(init_gene_emb, p=2, dim=-1)
    hyperedge_emb_norm = F.normalize(init_hyperedge_emb, p=2, dim=-1)
    init_partition = gene_emb_norm @ hyperedge_emb_norm.T

    return PCAInitialization(
        init_gene_emb=init_gene_emb,
        init_hyperedge_emb=init_hyperedge_emb,
        init_partition=init_partition,
        representative_indices=reps,
    )


def prepare_supervision(
    *,
    pos_mask,
    neg_mask,
    partial_pos_mask=None,
    pos_strength: float = 0.5,
    partial_pos_strength: float = 0.5,
    neg_strength: float = 0.0,
    dropout_rate: float = 0.0,
) -> tuple[str, dict[str, torch.Tensor], dict[str, float]]:
    pos_mask = dropout_bool_mask(_as_bool_mask(pos_mask, name="pos_mask"), dropout_rate)
    neg_mask = dropout_bool_mask(_as_bool_mask(neg_mask, name="neg_mask"), dropout_rate)

    if partial_pos_mask is None:
        _infer_num_genes(pos_mask, neg_mask)
        supervision_mode = "binary"
        masks = {"pos": pos_mask, "neg": neg_mask}
        targets = {"pos": float(pos_strength), "neg": float(neg_strength)}
        resolve_relation_supervision(
            supervision_mode="binary",
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            pos_threshold=pos_strength,
            neg_target=neg_strength,
        )
    else:
        partial_pos_mask = dropout_bool_mask(
            _as_bool_mask(partial_pos_mask, name="partial_pos_mask"),
            dropout_rate,
        )
        _infer_num_genes(pos_mask, neg_mask, partial_pos_mask)
        supervision_mode = "ternary"
        masks = {"full_pos": pos_mask, "partial_pos": partial_pos_mask, "neg": neg_mask}
        targets = {
            "full_pos": float(pos_strength),
            "partial_pos": float(partial_pos_strength),
            "neg": float(neg_strength),
        }
        resolve_relation_supervision(
            supervision_mode="ternary",
            full_pos_mask=pos_mask,
            partial_pos_mask=partial_pos_mask,
            neg_mask=neg_mask,
            full_pos_threshold=pos_strength,
            partial_pos_threshold=partial_pos_strength,
            neg_target=neg_strength,
        )

    return supervision_mode, masks, targets


def resolve_device(device: str | torch.device = "auto") -> str:
    if str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def run_supervised_hyperedges(
    *,
    pos_mask,
    neg_mask,
    partial_pos_mask=None,
    adata_path: str | Path | None = None,
    adata=None,
    num_genes: int | None = None,
    embed_dim: int = 16,
    num_hyperedges: int = 7,
    use_unassigned_hyperedge: bool = True,
    pos_strength: float = 0.5,
    partial_pos_strength: float = 0.5,
    neg_strength: float = 0.0,
    epochs: int = 10000,
    lr: float = 0.016,
    entropy_strength: float = 0.001,
    dropout_rate: float = 0.0,
    device: str | torch.device = "auto",
    ranges_map: Mapping[int, Sequence[tuple[int, int]]] | None = DEFAULT_HYPEREDGE_RANGES,
    representative_indices: Sequence[int] | None = None,
    seed: int | None = None,
) -> SupervisedHypergraphResult:
    if seed is not None:
        torch.manual_seed(seed)

    pos_mask = _as_bool_mask(pos_mask, name="pos_mask")
    neg_mask = _as_bool_mask(neg_mask, name="neg_mask")
    partial_pos_mask = (
        None
        if partial_pos_mask is None
        else _as_bool_mask(partial_pos_mask, name="partial_pos_mask")
    )
    inferred_num_genes = _infer_num_genes(pos_mask, neg_mask, partial_pos_mask)
    if num_genes is None:
        num_genes = inferred_num_genes
    elif num_genes != inferred_num_genes:
        raise ValueError(
            f"num_genes={num_genes} does not match mask size {inferred_num_genes}."
        )

    supervision_mode, masks, targets = prepare_supervision(
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        partial_pos_mask=partial_pos_mask,
        pos_strength=pos_strength,
        partial_pos_strength=partial_pos_strength,
        neg_strength=neg_strength,
        dropout_rate=dropout_rate,
    )
    supervised_gene_mask = supervised_gene_mask_from_relations(masks)
    unsupervised_gene_mask = ~supervised_gene_mask

    if use_unassigned_hyperedge:
        if num_hyperedges < 2:
            raise ValueError("num_hyperedges must be at least 2 when use_unassigned_hyperedge=True.")
        num_gene_modules = num_hyperedges - 1
    else:
        num_gene_modules = num_hyperedges

    if (
        use_unassigned_hyperedge
        and representative_indices is not None
        and len(representative_indices) == num_hyperedges
    ):
        representative_indices = representative_indices[:num_gene_modules]

    init = build_pca_initialization(
        adata_path=adata_path,
        adata=adata,
        num_genes=num_genes,
        embed_dim=embed_dim,
        num_hyperedges=num_gene_modules,
        ranges_map=ranges_map,
        representative_indices=representative_indices,
    )

    selected_device = resolve_device(device)
    train_kwargs = {
        "init_partition": init.init_partition,
        "init_hyperedge_emb": init.init_hyperedge_emb,
        "neg_mask": masks["neg"],
        "epochs": epochs,
        "lr": lr,
        "device": selected_device,
        "alpha": entropy_strength,
        "neg_target": neg_strength,
        "supervision_mode": supervision_mode,
        "entropy_gene_mask": supervised_gene_mask,
    }
    if supervision_mode == "binary":
        train_kwargs.update(
            {
                "pos_mask": masks["pos"],
                "pos_threshold": pos_strength,
            }
        )
    else:
        train_kwargs.update(
            {
                "full_pos_mask": masks["full_pos"],
                "partial_pos_mask": masks["partial_pos"],
                "full_pos_threshold": pos_strength,
                "partial_pos_threshold": partial_pos_strength,
            }
        )

    partition, hyperedge_emb, gene_emb, losses = train_embedding_weakpos0(**train_kwargs)

    null_hyperedge_index = None
    init_partition = init.init_partition
    init_hyperedge_emb = init.init_hyperedge_emb
    if use_unassigned_hyperedge:
        partition, hyperedge_emb, gene_emb, null_hyperedge_index = append_unassigned_hyperedge(
            partition_meaningful=partition,
            hyperedge_emb_meaningful=hyperedge_emb,
            supervised_gene_mask=supervised_gene_mask,
        )
        init_partition, init_hyperedge_emb, _, _ = append_unassigned_hyperedge(
            partition_meaningful=init.init_partition.to(partition.device),
            hyperedge_emb_meaningful=init.init_hyperedge_emb.to(hyperedge_emb.device),
            supervised_gene_mask=supervised_gene_mask,
        )

    return SupervisedHypergraphResult(
        partition=partition,
        hyperedge_emb=hyperedge_emb,
        gene_emb=gene_emb,
        losses=losses,
        init_gene_emb=init.init_gene_emb,
        init_hyperedge_emb=init_hyperedge_emb,
        init_partition=init_partition,
        representative_indices=init.representative_indices,
        supervision_mode=supervision_mode,
        relation_targets=targets,
        num_hyperedges=num_hyperedges,
        num_gene_modules=num_gene_modules,
        null_hyperedge_index=null_hyperedge_index,
        supervised_gene_mask=supervised_gene_mask.to(partition.device).detach(),
        unsupervised_gene_mask=unsupervised_gene_mask.to(partition.device).detach(),
        device=selected_device,
    )


def make_demo_binary_masks(
    num_genes: int = 420,
    *,
    dropout_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_genes < 420:
        raise ValueError("The demo masks require num_genes >= 420.")

    pos_mask = torch.zeros([num_genes, num_genes], dtype=torch.bool)
    neg_mask = torch.zeros([num_genes, num_genes], dtype=torch.bool)

    pos_mask[200:230, 200:230] = 1
    pos_mask[230:260, 230:260] = 1
    pos_mask[360:390, 360:390] = 1
    pos_mask[390:420, 390:420] = 1
    pos_mask[100:200, 100:200] = 1
    pos_mask[260:360, 260:360] = 1
    pos_mask[100:200, 160:260] = 1
    pos_mask[160:260, 100:200] = 1
    pos_mask[260:360, 360:420] = 1
    pos_mask[360:420, 260:360] = 1

    neg_mask[200:230, 230:260] = 1
    neg_mask[230:260, 200:230] = 1
    neg_mask[360:390, 390:420] = 1
    neg_mask[390:420, 360:390] = 1
    neg_mask[100:260, 260:420] = 1
    neg_mask[260:420, 100:260] = 1

    pos_mask[185:200, 185:260] = 0
    pos_mask[185:260, 185:200] = 0
    pos_mask[215:230, 215:230] = 0
    pos_mask[245:260, 245:260] = 0

    neg_mask[215:230, 245:260] = 0
    neg_mask[245:260, 215:230] = 0

    return dropout_bool_mask(pos_mask, dropout_rate), dropout_bool_mask(neg_mask, dropout_rate)


def make_demo_ternary_masks(
    num_genes: int = 420,
    *,
    dropout_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_genes < 420:
        raise ValueError("The demo masks require num_genes >= 420.")

    full_pos_mask = torch.zeros([num_genes, num_genes], dtype=torch.bool)
    partial_pos_mask = torch.zeros([num_genes, num_genes], dtype=torch.bool)
    neg_mask = torch.zeros([num_genes, num_genes], dtype=torch.bool)

    full_pos_mask[200:230, 200:230] = 1
    full_pos_mask[230:260, 230:260] = 1
    full_pos_mask[360:390, 360:390] = 1
    full_pos_mask[390:420, 390:420] = 1
    full_pos_mask[100:200, 100:200] = 1
    full_pos_mask[260:360, 260:360] = 1

    partial_pos_mask[100:200, 200:260] = 1
    partial_pos_mask[200:260, 100:200] = 1
    partial_pos_mask[260:360, 360:420] = 1
    partial_pos_mask[360:420, 260:360] = 1

    neg_mask[200:230, 230:260] = 1
    neg_mask[230:260, 200:230] = 1
    neg_mask[360:390, 390:420] = 1
    neg_mask[390:420, 360:390] = 1
    neg_mask[100:260, 260:420] = 1
    neg_mask[260:420, 100:260] = 1

    return (
        dropout_bool_mask(full_pos_mask, dropout_rate),
        dropout_bool_mask(partial_pos_mask, dropout_rate),
        dropout_bool_mask(neg_mask, dropout_rate),
    )


def plot_supervision_masks(pos_mask, neg_mask, partial_pos_mask=None):
    import matplotlib.pyplot as plt

    matrices = [(pos_mask, "pos_mask", "Reds")]
    if partial_pos_mask is not None:
        matrices.append((partial_pos_mask, "partial_pos_mask", "Oranges"))
    matrices.append((neg_mask, "neg_mask", "Blues"))

    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    if len(matrices) == 1:
        axes = [axes]

    for ax, (mat, title, cmap) in zip(axes, matrices):
        if not torch.is_tensor(mat):
            mat = torch.as_tensor(mat)
        image = ax.imshow(mat.detach().cpu().numpy(), cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Gene Index")
        ax.set_ylabel("Gene Index")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def summarize_unassigned_genes(result: SupervisedHypergraphResult) -> dict[str, int | None]:
    return {
        "num_gene_modules": result.num_gene_modules,
        "num_hyperedges_total": result.num_hyperedges,
        "null_hyperedge_index": result.null_hyperedge_index,
        "supervised_genes": int(result.supervised_gene_mask.sum().item()),
        "unassigned_genes": int(result.unsupervised_gene_mask.sum().item()),
    }


def plot_run_summary(result: SupervisedHypergraphResult):
    import matplotlib.pyplot as plt

    init_partition = result.init_partition.to(result.gene_emb.device)
    init_hyperedge_emb = result.init_hyperedge_emb.to(result.gene_emb.device)
    init_gene_emb = init_partition @ init_hyperedge_emb

    init_sim = init_gene_emb @ init_gene_emb.T
    trained_sim = result.gene_emb @ result.gene_emb.T
    hyperedge_sim = result.hyperedge_emb @ result.hyperedge_emb.T

    fig = plt.figure(figsize=(20, 14))
    panels = [
        (init_partition, "Init Partition (Raw)", "YlOrRd"),
        (init_hyperedge_emb, "Init Hyperedge Embedding", "viridis"),
        (init_gene_emb, "Init Gene Embedding", "viridis"),
        (result.partition, "Trained Partition", "YlOrRd"),
        (result.hyperedge_emb, "Trained Hyperedge Embedding", "viridis"),
        (result.gene_emb, "Trained Gene Embedding", "viridis"),
        (init_sim, "Init Gene Similarity (Dot Product)", "RdYlBu_r"),
        (trained_sim, "Trained Gene Similarity", "RdYlBu_r"),
        (hyperedge_sim, "Hyperedge Similarity", "RdYlBu_r"),
    ]

    for idx, (mat, title, cmap) in enumerate(panels, start=1):
        ax = plt.subplot(3, 3, idx)
        image = ax.imshow(mat.detach().cpu().numpy(), cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        if idx == 9:
            ax.set_xlabel("Hyperedge Index")
            ax.set_ylabel("Hyperedge Index")
        plt.colorbar(image, ax=ax)

    plt.tight_layout()
    return fig
