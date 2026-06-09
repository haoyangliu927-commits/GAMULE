from pathlib import Path
import json
import re
import sys

import matplotlib.pyplot as plt
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
from src.cme_visualization import (
    plot_combined_cme_supervision_heatmaps,
)
from src.metagene_tree import (
    aggregate_module_inclusion_from_genes,
    build_metagene_tree_from_result,
    load_cell_type_parent_map_from_xml,
    make_xml_parent_extractor,
    plot_module_inclusion_heatmaps,
    plot_module_inclusion_hierarchy,
    plot_metagene_tree,
    score_cell_types_from_metagene_tree,
    score_cell_hierarchy_from_cell_types,
    validate_metagene_tree_result,
)
from src.supervision_pipeline import (
    plot_run_summary,
    plot_supervision_masks,
    run_supervised_hyperedges,
    summarize_unassigned_genes,
)


# %%
ADATA_PATH = REPO_ROOT / "datasets/adata_303.h5ad"
tree_id_match = re.search(r"adata_(\d+)", ADATA_PATH.stem)
TREE_XML_PATH = (
    REPO_ROOT.parents[1] / "sc_simulator-main" / "sim_data" / "trees" / f"tree_{tree_id_match.group(1)}.xml"
    if tree_id_match
    else None
)
RESULT_PREFIX = ADATA_PATH.stem
RESULT_DIR = REPO_ROOT / "results" / f"{RESULT_PREFIX}_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read(ADATA_PATH)
num_cells = int(adata.n_obs)
num_genes = int(adata.n_vars)

print({"adata_path": str(ADATA_PATH), "num_cells": num_cells, "num_genes": num_genes})

cell_type_parent_map = {}
cell_type_parent_extractor = None
if TREE_XML_PATH is not None and TREE_XML_PATH.exists():
    cell_type_parent_map = load_cell_type_parent_map_from_xml(TREE_XML_PATH)
    cell_type_parent_extractor = make_xml_parent_extractor(cell_type_parent_map)
    pd.DataFrame(
        [
            {"cell_type": child, "parent_cell_type": parent}
            for child, parent in cell_type_parent_map.items()
        ]
    ).to_csv(RESULT_DIR / "cell_type_parent_map.csv", index=False)
    print({"tree_xml_path": str(TREE_XML_PATH), "cell_type_parent_map": cell_type_parent_map})
else:
    print(f"Tree XML not found, skip cell type and hierarchy accuracy: {TREE_XML_PATH}")

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

num_gene_modules = 8
num_gene_modules = max(1, min(num_gene_modules, num_genes))
num_hyperedges = num_gene_modules + 1

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
    "num_gene_modules": num_gene_modules,
    "num_hyperedges": num_hyperedges,
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

fig = plot_supervision_masks(pos_mask, neg_mask, partial_pos_mask=partial_pos_mask)
fig.savefig(RESULT_DIR / "supervision_masks.png", dpi=180, bbox_inches="tight")
plt.close(fig)


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
    use_unassigned_hyperedge=True,
    pos_strength=jaccard_pos_strength,
    partial_pos_strength=inclusion_partial_strength,
    neg_strength=neg_strength,
    hierarchy_strength=hierarchy_strength,
    hierarchy_margin=hierarchy_margin,
    hierarchy_min_direction_weight=hierarchy_min_direction_weight,
    epochs=1000,
    lr=0.016,
    entropy_strength=0.001,
    ranges_map=None,
    device="auto",
    seed=seed,
)

{
    "supervision_mode": result.supervision_mode,
    "relation_targets": result.relation_targets,
}

summarize_unassigned_genes(result)

fig = plot_run_summary(result)
fig.savefig(RESULT_DIR / "hyperedge_run_summary.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# %%
# ===== 超边 metagene CME -> 互斥图补图 -> 度优先树 =====
metagene_tree = build_metagene_tree_from_result(
    result,
    expression.matrix,
    adata=adata,
    gene_names=expression.gene_names,
    reference_column="gene_module",
    cme_threshold=t_CME,
    assignment="argmax",
    aggregation="sum",
    child_strategy="max_degree_ties",
)

tree_validation = validate_metagene_tree_result(metagene_tree)
assert tree_validation["tree_has_all_nodes"]
assert tree_validation["tree_edge_count_ok"]
assert tree_validation["root_is_empty_node"]

overall_accuracy = metagene_tree.stats.get("weighted_assignment_accuracy")
overall_accuracy_summary = {
    "overall_accuracy_percent": None if overall_accuracy is None else f"{overall_accuracy:.2%}",
    "correct_genes": metagene_tree.stats.get("assignment_total_correct"),
    "evaluable_genes": metagene_tree.stats.get("assignment_total_evaluable"),
}

from IPython.display import display, Markdown

acc = overall_accuracy_summary["overall_accuracy_percent"]
display(Markdown(f"## ✅ 基因模块识别准确率：**{acc}**"))

display(overall_accuracy_summary)
display(metagene_tree.stats)
display(pd.DataFrame(metagene_tree.module_assignment_table))
display(tree_validation)

module_rows = []
for node_index, (hyperedge, genes) in enumerate(metagene_tree.module_gene_names.items()):
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
pd.DataFrame(metagene_tree.module_assignment_table).to_csv(
    RESULT_DIR / "module_assignment_table.csv",
    index=False,
)
pd.DataFrame(
    [
        {"parent": int(parent), "child": int(child)}
        for parent, child in metagene_tree.tree_edges
    ]
).to_csv(RESULT_DIR / "tree_edges.csv", index=False)

fig = plot_metagene_tree(
    metagene_tree,
    save_path=RESULT_DIR / "metagene_cme_tree.png",
)
plt.close(fig)

module_inclusion = aggregate_module_inclusion_from_genes(
    metagene_tree,
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
pd.DataFrame(module_inclusion.edge_table).to_csv(
    RESULT_DIR / "module_inclusion_edges.csv",
    index=False,
)
pd.DataFrame(module_inclusion.selected_edge_table).to_csv(
    RESULT_DIR / "module_inclusion_selected_edges.csv",
    index=False,
)

display(Markdown("## Module-module directed inclusion"))
display(module_inclusion.stats)
display(pd.DataFrame(module_inclusion.selected_edge_table))

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


# %%
# ===== metagene score -> cell type 判断 =====
cell_type_accuracy_summary = None
hierarchy_accuracy_summary = None
if "cell_type" in adata.obs and cell_type_parent_extractor is not None:
    cell_type_result = score_cell_types_from_metagene_tree(
        metagene_tree,
        adata=adata,
        obs_column="cell_type",
    )
    cell_type_accuracy_summary = {
        "cell_type_accuracy": cell_type_result.stats["cell_type_accuracy"],
        "cell_type_accuracy_percent": f"{cell_type_result.stats['cell_type_accuracy']:.2%}",
        "correct_cells": cell_type_result.stats["cell_type_total_correct"],
        "evaluable_cells": cell_type_result.stats["cell_type_total_evaluable"],
    }
    acc_pct = f"{cell_type_result.stats['cell_type_accuracy']:.2%}"
    display(Markdown(f"## ✅ 细胞类型识别准确率：**{acc_pct}**"))
    display(cell_type_accuracy_summary)
    display(pd.DataFrame(cell_type_result.metagene_cell_type_table))
    pd.DataFrame(cell_type_result.metagene_cell_type_table).to_csv(
        RESULT_DIR / "metagene_cell_type_table.csv",
        index=False,
    )
    pd.DataFrame(cell_type_result.cell_assignment_table).to_csv(
        RESULT_DIR / "cell_assignment_table.csv",
        index=False,
    )

    # ===== subtype -> parent type 从属关系判断 =====
    hierarchy_result = score_cell_hierarchy_from_cell_types(
        cell_type_result,
        parent_extractor=cell_type_parent_extractor,
    )
    hierarchy_accuracy_summary = {
        "hierarchy_accuracy": hierarchy_result.stats["hierarchy_accuracy"],
        "hierarchy_accuracy_percent": f"{hierarchy_result.stats['hierarchy_accuracy']:.2%}",
        "correct_cells": hierarchy_result.stats["hierarchy_total_correct"],
        "evaluable_cells": hierarchy_result.stats["hierarchy_total_evaluable"],
    }
    hierarchy_acc_pct = f"{hierarchy_result.stats['hierarchy_accuracy']:.2%}"
    display(Markdown(f"## ✅ 细胞从属关系识别准确率：**{hierarchy_acc_pct}**"))
    display(hierarchy_accuracy_summary)
    display(pd.DataFrame(hierarchy_result.hierarchy_relation_table))
    pd.DataFrame(hierarchy_result.hierarchy_relation_table).to_csv(
        RESULT_DIR / "hierarchy_relation_table.csv",
        index=False,
    )
else:
    if "cell_type" not in adata.obs:
        print("adata.obs 里没有 'cell_type' 列，跳过细胞类型和从属关系准确率计算。")
    else:
        print("没有可用的 tree XML，跳过细胞类型和从属关系准确率计算。")

summary = {
    "adata_path": str(ADATA_PATH),
    "tree_xml_path": str(TREE_XML_PATH) if TREE_XML_PATH is not None else None,
    "cell_type_parent_map": cell_type_parent_map,
    "shape": list(adata.shape),
    "result_dir": str(RESULT_DIR),
    "run_config": run_config,
    "supervision_stats": supervision.stats,
    "mask_summary": mask_entry_summary,
    "unassigned_summary": summarize_unassigned_genes(result),
    "metagene_tree_stats": metagene_tree.stats,
    "module_inclusion_stats": module_inclusion.stats,
    "tree_validation": tree_validation,
    "gene_module_accuracy": overall_accuracy_summary,
    "cell_type_accuracy": cell_type_accuracy_summary,
    "hierarchy_accuracy": hierarchy_accuracy_summary,
    "outputs": {
        "input_expression_heatmap": "input_expression_heatmap.png",
        "combined_supervision_heatmaps": "combined_supervision_heatmaps.png",
        "supervision_masks": "supervision_masks.png",
        "hyperedge_run_summary": "hyperedge_run_summary.png",
        "metagene_cme_tree": "metagene_cme_tree.png",
        "module_inclusion_heatmaps": "module_inclusion_heatmaps.png",
        "module_inclusion_hierarchy": "module_inclusion_hierarchy.png",
        "module_inclusion_edges": "module_inclusion_edges.csv",
        "module_inclusion_selected_edges": "module_inclusion_selected_edges.csv",
        "gene_modules": "gene_modules.csv",
        "tree_edges": "tree_edges.csv",
    },
}
with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

print({"result_dir": str(RESULT_DIR), "summary": str(RESULT_DIR / "summary.json")})


# %%
