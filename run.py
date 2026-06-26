from pathlib import Path
import json
import re
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy import sparse


print(sys.executable)

candidates = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path.cwd(),
    Path.cwd().parent
]
REPO_ROOT = next(
    path for path in candidates
    if (path / "src/cme_supervision.py").exists() and (path / "datasets").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.cme_supervision import (
    build_supervision_from_cme,
    compute_cme_matrix,
    compute_cme_pvalues,
    load_expression_from_h5ad,
    save_supervision_npz,
)
from src.gene_ambiguity import compute_gene_ambiguity
from src.cme_visualization import (
    plot_combined_cme_supervision_heatmaps,
)
from src.metagene_tree import (
    aggregate_module_inclusion_from_genes,
    genes_by_hyperedge,
    plot_module_inclusion_heatmaps,
    plot_module_inclusion_hierarchy,
)
from src.supervision_pipeline import (
    plot_loss_history,
    plot_run_summary,
    run_supervised_hyperedges,
    summarize_unassigned_genes,
)


# %%
ADATA_PATH = REPO_ROOT / "datasets/adata_ngn3_ss.h5ad"
RESULT_PREFIX = ADATA_PATH.stem
RESULT_DIR = REPO_ROOT / "results" / f"{RESULT_PREFIX}_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read(ADATA_PATH)
num_cells = int(adata.n_obs)
num_genes = int(adata.n_vars)

print({"adata_path": str(ADATA_PATH), "num_cells": num_cells, "num_genes": num_genes})

def get_gene_module_labels(adata, col_name="gene_module"):
    pattern = re.compile(r"^g(?P<module>[A-Za-z0-9]+)_gb\d+_\d+$")
    labels = []
    for gene in adata.var_names.astype(str):
        match = pattern.match(gene)
        labels.append(match.group("module") if match else "NA")
    return pd.Series(labels, index=adata.var_names, name=col_name)

adata.var["gene_module"] = get_gene_module_labels(adata)
observed_gene_modules = sorted(
    label for label in adata.var["gene_module"].unique().tolist()
    if label not in {"NA", ""}
)
observed_cell_types = sorted(adata.obs["cell_type"].astype(str).unique().tolist()) if "cell_type" in adata.obs else []

print({
    "observed_gene_modules": observed_gene_modules,
    "observed_cell_types": observed_cell_types,
})

plot_x = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
ax = sns.heatmap(plot_x)
ax.set_xlabel("Genes", fontsize=12)
ax.set_ylabel("Cells", fontsize=12)
ax.figure.savefig(RESULT_DIR / "input_expression_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(ax.figure)

# %%
seed = 0

t_CME = 0.66
t_p = 0.05
t_Jaccard = 0.50
k_Jaccard = 297
t_inclusion = 0.10
jaccard_pos_strength = 0.5
inclusion_partial_strength = 0.25
neg_strength = 0.0
hierarchy_strength = 0.0
hierarchy_margin = 0.0
hierarchy_min_direction_weight = 0.0
n_permutations = 100

num_biological_modules = 3
num_garbage_modules = 1
num_hyperedges = num_biological_modules + num_garbage_modules
num_hyperedges = max(1, min(num_hyperedges, num_genes))
garbage_hyperedge_index = num_hyperedges - 1
garbage_strength = 0.05
clean_repel_strength = 0.0
garbage_margin_strength = 0.05
garbage_margin = 0.1
exclude_garbage_from_relation_loss = True
initialize_all_to_garbage = True
garbage_init_logit = 0.2
normal_init_logit = 0.0
use_unassigned_hyperedge = False
training_epochs = 10000
entropy_strength = 0.001
entropy_schedule = "delayed_linear"
entropy_warmup_start_fraction = 0.5
entropy_warmup_end_fraction = 1.0

torch.manual_seed(seed)

run_config = {
    "seed": seed,
    "t_CME": t_CME,
    "t_p": t_p,
    "t_Jaccard": t_Jaccard,
    "k_Jaccard": k_Jaccard,
    "t_inclusion": t_inclusion,
    "jaccard_pos_strength": jaccard_pos_strength,
    "inclusion_partial_strength": inclusion_partial_strength,
    "neg_strength": neg_strength,
    "hierarchy_strength": hierarchy_strength,
    "hierarchy_margin": hierarchy_margin,
    "hierarchy_min_direction_weight": hierarchy_min_direction_weight,
    "n_permutations": n_permutations,
    "num_biological_modules": num_biological_modules,
    "num_garbage_modules": num_garbage_modules,
    "num_hyperedges": num_hyperedges,
    "garbage_hyperedge_index": garbage_hyperedge_index,
    "garbage_strength": garbage_strength,
    "clean_repel_strength": clean_repel_strength,
    "garbage_margin_strength": garbage_margin_strength,
    "garbage_margin": garbage_margin,
    "exclude_garbage_from_relation_loss": exclude_garbage_from_relation_loss,
    "initialize_all_to_garbage": initialize_all_to_garbage,
    "garbage_init_logit": garbage_init_logit,
    "normal_init_logit": normal_init_logit,
    "use_unassigned_hyperedge": use_unassigned_hyperedge,
    "training_epochs": training_epochs,
    "entropy_strength": entropy_strength,
    "entropy_schedule": entropy_schedule,
    "entropy_warmup_start_fraction": entropy_warmup_start_fraction,
    "entropy_warmup_end_fraction": entropy_warmup_end_fraction,
}
run_config


# %%
expression = load_expression_from_h5ad(
    ADATA_PATH,
    num_genes=num_genes,
)
cme_matrix = compute_cme_matrix(expression.matrix, normalize=False, use_numba=True)
pval_matrix = compute_cme_pvalues(
    expression.matrix,
    cme_matrix,
    n_permutations=n_permutations,
    seed=seed,
    use_numba=True,
)

supervision = build_supervision_from_cme(
    cme_matrix,
    pval_matrix,
    cme_threshold=t_CME,
    pvalue_threshold=t_p,
    jaccard_threshold=t_Jaccard,
    top_k=k_Jaccard,
    inclusion_threshold=t_inclusion,
)

pos_mask = torch.from_numpy(supervision.positive_mask)
neg_mask = torch.from_numpy(supervision.negative_mask)
partial_pos_mask = (
    None
    if supervision.inclusion_partial_mask is None
    else torch.from_numpy(supervision.inclusion_partial_mask)
)
directed_inclusion_mask = (
    None
    if supervision.inclusion_directed_mask is None
    else torch.from_numpy(supervision.inclusion_directed_mask)
)

save_supervision_npz(
    RESULT_DIR / "three_way_supervision_matrices.npz",
    cme_matrix=cme_matrix,
    pval_matrix=pval_matrix,
    supervision=supervision,
)

if partial_pos_mask is None:
    raise RuntimeError("inclusion_threshold was set, but inclusion_partial_mask was not generated.")
if directed_inclusion_mask is None:
    raise RuntimeError("inclusion_threshold was set, but inclusion_directed_mask was not generated.")
supervision.stats

ambiguity = compute_gene_ambiguity(
    positive_mask=supervision.positive_mask,
    negative_mask=supervision.negative_mask,
    partial_pos_mask=supervision.inclusion_partial_mask,
    directed_inclusion_mask=supervision.inclusion_directed_mask,
    gene_names=expression.gene_names,
)
ambiguity_table = ambiguity.table.copy()
if "gene_module" in adata.var:
    ambiguity_table["reference_gene_module"] = adata.var["gene_module"].astype(str).values[
        : len(ambiguity_table)
    ]
ambiguity_table.to_csv(RESULT_DIR / "gene_ambiguity_scores.csv", index=False)

mask_entry_summary = {
    "full_pos": int(pos_mask.sum().item()),
    "partial_pos": int(partial_pos_mask.sum().item()),
    "directed_inclusion": int(directed_inclusion_mask.sum().item()),
    "neg": int(neg_mask.sum().item()),
}
mask_entry_summary

_ = plot_combined_cme_supervision_heatmaps(
    cme_matrix=cme_matrix,
    pval_matrix=pval_matrix,
    supervision=supervision,
    save_path=RESULT_DIR / "combined_supervision_heatmaps.png",
    show=False,
);


# %%
# ===== 三路监督 -> 超边 gene modules =====
result = run_supervised_hyperedges(
    adata=adata,
    pos_mask=pos_mask,
    neg_mask=neg_mask,
    partial_pos_mask=partial_pos_mask,
    directed_inclusion_mask=directed_inclusion_mask,
    num_genes=num_genes,
    num_hyperedges=num_hyperedges,
    use_unassigned_hyperedge=use_unassigned_hyperedge,
    pos_strength=jaccard_pos_strength,
    partial_pos_strength=inclusion_partial_strength,
    neg_strength=neg_strength,
    hierarchy_strength=hierarchy_strength,
    hierarchy_margin=hierarchy_margin,
    hierarchy_min_direction_weight=hierarchy_min_direction_weight,
    initialize_all_to_garbage=initialize_all_to_garbage,
    garbage_init_logit=garbage_init_logit,
    normal_init_logit=normal_init_logit,
    garbage_hyperedge_index=garbage_hyperedge_index,
    ambiguity_weight=torch.from_numpy(ambiguity.ambiguity_weight),
    garbage_strength=garbage_strength,
    clean_repel_strength=clean_repel_strength,
    garbage_margin_strength=garbage_margin_strength,
    garbage_margin=garbage_margin,
    exclude_garbage_from_relation_loss=exclude_garbage_from_relation_loss,
    epochs=training_epochs,
    lr=0.016,
    entropy_strength=entropy_strength,
    entropy_schedule=entropy_schedule,
    entropy_warmup_start_fraction=entropy_warmup_start_fraction,
    entropy_warmup_end_fraction=entropy_warmup_end_fraction,
    ranges_map=None,
    device="auto",
    seed=seed,
)

{
    "supervision_mode": result.supervision_mode,
    "relation_targets": result.relation_targets,
}

summarize_unassigned_genes(result)

loss_history_df = pd.DataFrame(result.loss_history)
loss_history_df.to_csv(RESULT_DIR / "training_loss_history.csv", index=False)
fig = plot_loss_history(result, save_path=RESULT_DIR / "training_loss_history.png")
plt.close(fig)

partition_np = result.partition.detach().cpu().numpy()
assigned_hyperedge = partition_np.argmax(axis=1).astype(int)
garbage_probability = partition_np[:, garbage_hyperedge_index]
normal_hyperedge_indices = [
    idx for idx in range(num_hyperedges) if idx != garbage_hyperedge_index
]
normal_probabilities = partition_np[:, normal_hyperedge_indices]
max_normal_probability = normal_probabilities.max(axis=1)
normal_prob_sum = normal_probabilities.sum(axis=1, keepdims=True)
normal_probabilities_for_entropy = normal_probabilities / np.maximum(normal_prob_sum, 1e-8)
assignment_entropy_normal = -np.sum(
    normal_probabilities_for_entropy * np.log(normal_probabilities_for_entropy + 1e-8),
    axis=1,
) / np.log(max(2, len(normal_hyperedge_indices)))
garbage_probability_margin = garbage_probability - max_normal_probability

gene_assignment_diagnostics = pd.DataFrame(
    {
        "gene_index": range(num_genes),
        "gene_name": expression.gene_names[:num_genes],
        "assigned_hyperedge": assigned_hyperedge,
        "garbage_probability": garbage_probability,
        "max_normal_probability": max_normal_probability,
        "garbage_probability_minus_max_normal_probability": garbage_probability_margin,
        "assignment_entropy_normal": assignment_entropy_normal,
        "ambiguity_weight": ambiguity.ambiguity_weight,
        "robust_z": ambiguity.robust_z,
        "raw_ambiguity_score": ambiguity.raw_ambiguity_score,
        "contradiction_degree": ambiguity_table["contradiction_degree"].to_numpy(),
        "bidirectional_inclusion_degree": ambiguity_table[
            "bidirectional_inclusion_degree"
        ].to_numpy(),
    }
)
if "gene_module" in adata.var:
    gene_assignment_diagnostics["reference_gene_module"] = adata.var["gene_module"].astype(
        str
    ).values[:num_genes]
gene_assignment_diagnostics.to_csv(
    RESULT_DIR / "gene_assignment_diagnostics.csv",
    index=False,
)
gene_assignment_diagnostics.loc[
    gene_assignment_diagnostics["assigned_hyperedge"] == garbage_hyperedge_index
].to_csv(RESULT_DIR / "garbage_genes_only.csv", index=False)

fig = plot_run_summary(result)
fig.savefig(RESULT_DIR / "hyperedge_run_summary.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# %%
# ===== 训练后的超边 gene modules -> module-level directed inclusion =====
module_gene_indices = genes_by_hyperedge(
    result.partition,
    null_hyperedge_index=result.null_hyperedge_index,
    meaningful_hyperedge_indices=range(result.num_gene_modules),
    assignment="argmax",
)
module_gene_names = {
    module_idx: [expression.gene_names[int(idx)] for idx in gene_indices]
    for module_idx, gene_indices in module_gene_indices.items()
}
module_ids = list(module_gene_indices.keys())
node_labels = [f"H{module_id}\nn={len(module_gene_indices[module_id])}" for module_id in module_ids]
module_view = SimpleNamespace(
    module_gene_indices=module_gene_indices,
    module_gene_names=module_gene_names,
    module_ids=module_ids,
    node_labels=node_labels,
)
module_rows = []
for node_index, (hyperedge, genes) in enumerate(module_gene_names.items()):
    module_rows.append(
        {
            "node_index": int(node_index),
            "hyperedge": int(hyperedge),
            "num_genes": len(genes),
            "genes": ";".join(map(str, genes)),
        }
    )
module_df = pd.DataFrame(module_rows)
module_df.to_csv(RESULT_DIR / "gene_modules.csv", index=False)

module_inclusion = aggregate_module_inclusion_from_genes(
    module_view,
    supervision.inclusion_score_matrix,
    directed_inclusion_mask=supervision.inclusion_directed_mask,
    min_mean_score=t_inclusion,
    min_directed_fraction=0.0,
)
module_inclusion_labels = [f"H{module_id}" for module_id in module_inclusion.module_ids]
pd.DataFrame(
    module_inclusion.mean_score_matrix,
    index=module_inclusion_labels,
    columns=module_inclusion_labels,
).to_csv(RESULT_DIR / "module_inclusion_mean_score_matrix.csv")
pd.DataFrame(
    module_inclusion.directed_fraction_matrix,
    index=module_inclusion_labels,
    columns=module_inclusion_labels,
).to_csv(RESULT_DIR / "module_inclusion_directed_fraction_matrix.csv")
pd.DataFrame(module_inclusion.selected_edge_table).to_csv(
    RESULT_DIR / "module_inclusion_selected_edges.csv",
    index=False,
)

print({"module_inclusion_stats": module_inclusion.stats})
print(pd.DataFrame(module_inclusion.selected_edge_table))

fig = plot_module_inclusion_heatmaps(
    module_inclusion,
    save_path=RESULT_DIR / "module_inclusion_heatmaps.png",
)
plt.close(fig)
fig = plot_module_inclusion_hierarchy(
    module_inclusion,
    save_path=RESULT_DIR / "module_inclusion_hierarchy.png",
)
plt.close(fig)

assigned_garbage_mask = assigned_hyperedge == garbage_hyperedge_index
if assigned_garbage_mask.any():
    mean_garbage_probability_assigned_garbage = float(
        garbage_probability[assigned_garbage_mask].mean()
    )
    mean_ambiguity_weight_assigned_garbage = float(
        ambiguity.ambiguity_weight[assigned_garbage_mask].mean()
    )
    mean_max_normal_probability_assigned_garbage = float(
        max_normal_probability[assigned_garbage_mask].mean()
    )
    mean_garbage_margin_assigned_garbage = float(
        garbage_probability_margin[assigned_garbage_mask].mean()
    )
else:
    mean_garbage_probability_assigned_garbage = None
    mean_ambiguity_weight_assigned_garbage = None
    mean_max_normal_probability_assigned_garbage = None
    mean_garbage_margin_assigned_garbage = None

non_garbage_mask = ~assigned_garbage_mask
mean_ambiguity_weight_non_garbage = (
    float(ambiguity.ambiguity_weight[non_garbage_mask].mean())
    if non_garbage_mask.any()
    else None
)
if "reference_gene_module" in gene_assignment_diagnostics:
    reference_module_counts_assigned_garbage = (
        gene_assignment_diagnostics.loc[
            assigned_garbage_mask,
            "reference_gene_module",
        ]
        .value_counts()
        .to_dict()
    )
else:
    reference_module_counts_assigned_garbage = {}

summary = {
    "adata_path": str(ADATA_PATH),
    "shape": list(adata.shape),
    "result_dir": str(RESULT_DIR),
    "run_config": run_config,
    "supervision_stats": supervision.stats,
    "mask_summary": mask_entry_summary,
    "unassigned_summary": summarize_unassigned_genes(result),
    "garbage_hyperedge_index": int(garbage_hyperedge_index),
    "garbage_strength": float(garbage_strength),
    "clean_repel_strength": float(clean_repel_strength),
    "garbage_margin_strength": float(garbage_margin_strength),
    "garbage_margin": float(garbage_margin),
    "exclude_garbage_from_relation_loss": bool(exclude_garbage_from_relation_loss),
    "mean_ambiguity_weight": float(ambiguity.ambiguity_weight.mean()),
    "num_high_ambiguity_genes": int((ambiguity.ambiguity_weight > 0.5).sum()),
    "num_genes_assigned_to_garbage": int((assigned_hyperedge == garbage_hyperedge_index).sum()),
    "final_loss": float(result.losses[-1]) if result.losses else None,
    "min_loss": float(min(result.losses)) if result.losses else None,
    "min_loss_epoch": int(np.argmin(result.losses) + 1) if result.losses else None,
    "mean_garbage_probability_assigned_garbage": mean_garbage_probability_assigned_garbage,
    "mean_ambiguity_weight_assigned_garbage": mean_ambiguity_weight_assigned_garbage,
    "mean_ambiguity_weight_non_garbage": mean_ambiguity_weight_non_garbage,
    "mean_max_normal_probability_assigned_garbage": mean_max_normal_probability_assigned_garbage,
    "mean_garbage_margin_assigned_garbage": mean_garbage_margin_assigned_garbage,
    "reference_module_counts_assigned_garbage": reference_module_counts_assigned_garbage,
    "module_inclusion_stats": module_inclusion.stats,
    "outputs": {
        "input_expression_heatmap": "input_expression_heatmap.png",
        "combined_supervision_heatmaps": "combined_supervision_heatmaps.png",
        "hyperedge_run_summary": "hyperedge_run_summary.png",
        "training_loss_history": "training_loss_history.csv",
        "training_loss_history_plot": "training_loss_history.png",
        "gene_ambiguity_scores": "gene_ambiguity_scores.csv",
        "gene_assignment_diagnostics": "gene_assignment_diagnostics.csv",
        "garbage_genes_only": "garbage_genes_only.csv",
        "module_inclusion_heatmaps": "module_inclusion_heatmaps.png",
        "module_inclusion_hierarchy": "module_inclusion_hierarchy.png",
        "module_inclusion_selected_edges": "module_inclusion_selected_edges.csv",
        "gene_modules": "gene_modules.csv",
    },
}
with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

print({"result_dir": str(RESULT_DIR), "summary": str(RESULT_DIR / "summary.json")})


# %%
