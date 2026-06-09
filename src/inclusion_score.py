#!/usr/bin/env python3
"""
Directional Inclusion Score (DIS)
=================================
量化两个基因之间的"有向包含关系"：
1. expression mode: 基因 B 是否为基因 A 的亚型。
2. mutex-profile mode: 基因 A 的互斥关系集合是否被基因 B 覆盖。

公式: Score = Asymmetry × Co-occurrence^0.5 × Difference^0.5

三个过滤器:
  1. 方向性: 1 - min(#[1,0],#[0,1]) / max(#[1,0],#[0,1])  → 要求一边倒
  2. 共存性: sqrt(#[1,1] / (#[1,1]+#[1,0]+#[0,1]))         → 要求有交集
  3. 差异性: sqrt((#[1,0]+#[0,1]) / (#[1,1]+#[1,0]+#[0,1])) → 要求不完全重合

输入: .h5ad 文件 (cells × genes)，或直接输入预计算的 CME/互斥分数矩阵
输出: 有向包含分数矩阵 + 层级分类 + 可视化

Usage:
    python inclusion_score.py input.h5ad
    python inclusion_score.py input.h5ad --threshold 1.0 --by-celltype
    python inclusion_score.py input.h5ad --cme-matrix cme.csv --top 50
    python inclusion_score.py cme.csv --mutex-profile --mutex-threshold 0.9
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from itertools import combinations
from collections import defaultdict

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Inclusion Score Computation
# ═══════════════════════════════════════════════════════════════════════════════

def binarize_matrix(X, threshold=1.0):
    """将连续表达矩阵二值化: > threshold → 1, else → 0"""
    return (X > threshold).astype(np.int8)


def compute_cooccurrence(gene_a, gene_b):
    """计算两个二值向量的共现统计: n11, n10, n01"""
    n11 = int(np.sum((gene_a == 1) & (gene_b == 1)))
    n10 = int(np.sum((gene_a == 1) & (gene_b == 0)))
    n01 = int(np.sum((gene_a == 0) & (gene_b == 1)))
    return n11, n10, n01


def _directed_inclusion_score(
    n11,
    violation,
    extra,
    *,
    direction_power=0.5,
    shared_power=0.5,
    difference_power=0.5,
):
    """
    Directional inclusion kernel.

    violation: 目标方向上不该出现的单边关系，必须远少于 extra。
    extra:     目标方向上允许出现的单边关系，用来提供层级差异。

    Score = max(0, 1 - (violation / extra)^0.5)  ← 方向性
          × (n11 / total)^0.5                    ← 共有关系不能太少
          × ((violation+extra) / total)^0.5      ← 但也不能完全重合
    """
    total = n11 + violation + extra
    if total == 0:
        return 0.0
    if n11 <= 0 or extra <= 0:
        return 0.0

    direction = 1.0 - (violation / extra) ** direction_power
    direction = max(0.0, direction)
    shared = (n11 / total) ** shared_power
    difference = ((violation + extra) / total) ** difference_power

    return direction * shared * difference


def inclusion_score(n11, n10, n01):
    """
    表达层面的有向包含分数 Score(A⊃B)。

    B 是 A 的亚型: B 出现时 A 应该出现，因此 [0,1] 是违反项；
    A 出现但 B 不出现的 [1,0] 是允许的 parent-only 差异。
    """
    return _directed_inclusion_score(n11, violation=n01, extra=n10)


def mutex_profile_inclusion_score(n11, n10, n01):
    """
    互斥关系指纹的有向包含分数 Score(A->B)。

    A 是更大类 marker，B 是亚型 marker 时：
    A 有的互斥关系 B 应该也有，因此 [1,0] 是违反项；
    B 额外拥有而 A 没有的 [0,1] 是允许的 child-specific 差异。
    """
    return _directed_inclusion_score(n11, violation=n10, extra=n01)


def compute_directed_score(gene_a, gene_b):
    """计算两个方向的包含分数 + 共现统计"""
    n11, n10, n01 = compute_cooccurrence(gene_a, gene_b)
    score_A_sup_B = inclusion_score(n11, n10, n01)
    score_B_sup_A = inclusion_score(n11, n01, n10)
    return score_A_sup_B, score_B_sup_A, n11, n10, n01


def compute_inclusion_matrix(X_binary, gene_names):
    """
    计算所有基因对的有向包含分数矩阵。

    返回:
    - score_matrix: (n, n), score_matrix[i,j] = Score(gene_i ⊃ gene_j)
    - detail_df: 每对基因的详细统计
    """
    n_genes = X_binary.shape[1]
    score_matrix = np.zeros((n_genes, n_genes))
    details = []

    for i, j in combinations(range(n_genes), 2):
        a, b = X_binary[:, i], X_binary[:, j]
        si, sj, n11, n10, n01 = compute_directed_score(a, b)

        score_matrix[i, j] = si
        score_matrix[j, i] = sj

        if si > 0.01 or sj > 0.01:
            details.append({
                "gene_A": gene_names[i], "gene_B": gene_names[j],
                "n11": n11, "n10": n10, "n01": n01,
                "score_A_sup_B": round(si, 4),
                "score_B_sup_A": round(sj, 4),
            })

    return score_matrix, pd.DataFrame(details) if details else pd.DataFrame(
        columns=["gene_A","gene_B","n11","n10","n01","score_A_sup_B","score_B_sup_A"])


def load_square_matrix(path, *, matrix_key="cme_matrix"):
    """Load a square gene-by-gene matrix from CSV/TSV/NPZ/NPY."""
    path = str(path)
    suffix = os.path.splitext(path)[1].lower()

    gene_names = None
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, index_col=0, sep=sep)
        matrix = df.to_numpy(dtype=np.float32)
        gene_names = [str(g) for g in df.index]
        if len(df.columns) == matrix.shape[1]:
            col_names = [str(g) for g in df.columns]
            if col_names != gene_names:
                print("  WARNING: matrix row names and column names differ; using row names.")
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if matrix_key not in data:
            available = ", ".join(data.files)
            raise ValueError(f"NPZ does not contain key {matrix_key!r}. Available keys: {available}")
        matrix = np.asarray(data[matrix_key], dtype=np.float32)
        if "gene_names" in data:
            gene_names = [str(g) for g in data["gene_names"].tolist()]
    elif suffix == ".npy":
        matrix = np.asarray(np.load(path), dtype=np.float32)
    else:
        raise ValueError("Matrix input must be one of .csv, .tsv, .txt, .npz, or .npy.")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {matrix.shape}.")
    if gene_names is None:
        gene_names = [f"gene_{idx}" for idx in range(matrix.shape[0])]
    if len(gene_names) != matrix.shape[0]:
        raise ValueError("gene_names length does not match matrix size.")
    return matrix, np.asarray(gene_names, dtype=object)


def binarize_mutex_matrix(cme_matrix, threshold=0.9):
    """Convert a CME/mutual-exclusion score matrix to a 0/1 relation matrix."""
    matrix = np.asarray(cme_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cme_matrix must be square, got shape {matrix.shape}.")
    binary = matrix >= threshold
    binary = binary | binary.T
    np.fill_diagonal(binary, False)
    return binary.astype(np.int8)


def compute_mutex_profile_counts(parent_profile, child_profile):
    """Counts for two binary mutual-exclusion profiles: n11, n10, n01."""
    parent_profile = np.asarray(parent_profile, dtype=bool)
    child_profile = np.asarray(child_profile, dtype=bool)
    n11 = int(np.sum(parent_profile & child_profile))
    n10 = int(np.sum(parent_profile & (~child_profile)))
    n01 = int(np.sum((~parent_profile) & child_profile))
    return n11, n10, n01


def compute_mutex_profile_inclusion_matrix(
    mutex_binary,
    gene_names,
    *,
    exclude_pair_entries=True,
):
    """
    Compare each pair of genes by their mutual-exclusion relation profiles.

    score_matrix[i,j] = Score(gene_i -> gene_j), meaning gene_i's mutual-exclusion
    relations are mostly covered by gene_j's relations.
    """
    binary = np.asarray(mutex_binary, dtype=bool)
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
        raise ValueError(f"mutex_binary must be square, got shape {binary.shape}.")

    n_genes = binary.shape[0]
    gene_names = list(gene_names)
    if len(gene_names) != n_genes:
        raise ValueError("gene_names length does not match mutex_binary size.")

    score_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
    details = []

    for i, j in combinations(range(n_genes), 2):
        profile_i = binary[i].copy()
        profile_j = binary[j].copy()
        if exclude_pair_entries:
            profile_i[[i, j]] = False
            profile_j[[i, j]] = False

        n11, n10, n01 = compute_mutex_profile_counts(profile_i, profile_j)
        si = mutex_profile_inclusion_score(n11, n10, n01)
        sj = mutex_profile_inclusion_score(n11, n01, n10)

        score_matrix[i, j] = si
        score_matrix[j, i] = sj

        if si > 0.01 or sj > 0.01:
            details.append({
                "gene_A": gene_names[i],
                "gene_B": gene_names[j],
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "score_A_sup_B": round(float(si), 4),
                "score_B_sup_A": round(float(sj), 4),
            })

    return score_matrix, pd.DataFrame(details) if details else pd.DataFrame(
        columns=["gene_A", "gene_B", "n11", "n10", "n01", "score_A_sup_B", "score_B_sup_A"])


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchical Classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_hierarchy(score_matrix, gene_names, min_score=0.15):
    """
    根据包含分数矩阵对基因进行层级分类。

    - Level 0 (Root): 包含最多其他基因的基因 → 大类 marker
    - Level 1: 被 Level 0 包含、同时包含其他基因的基因 → 亚型 marker
    - Level 2+: 更细的亚型 marker

    返回: DataFrame with gene, level, n_children, n_parents, role
    """
    n = len(gene_names)
    gene_names = list(gene_names)

    # 统计每个基因的入度(被多少基因包含)和出度(包含多少基因)
    in_degree = np.zeros(n)   # 被包含次数 (列为 parent)
    out_degree = np.zeros(n)  # 包含其他基因次数 (行为 parent)

    for i in range(n):
        for j in range(n):
            if i != j and score_matrix[i, j] >= min_score:
                out_degree[i] += 1  # i ⊃ j
                in_degree[j] += 1   # j 被 i 包含

    # 分类
    results = []
    for i, g in enumerate(gene_names):
        # 角色判定
        if out_degree[i] > 0 and in_degree[i] == 0:
            role = "root_marker"       # 只包含别人，不被包含 → 最高层
        elif out_degree[i] > 0 and in_degree[i] > 0:
            role = "intermediate"      # 既被包含又包含别人 → 中间层
        elif out_degree[i] == 0 and in_degree[i] > 0:
            role = "leaf_marker"       # 只被包含 → 最底层（最细亚型）
        else:
            role = "independent"       # 无包含关系

        # 层级: root=0, intermediate=1, leaf=2
        if role == "root_marker":
            level = 0
        elif role == "intermediate":
            level = 1
        elif role == "leaf_marker":
            level = 2
        else:
            level = -1

        results.append({
            "gene": g,
            "level": level,
            "role": role,
            "n_children": int(out_degree[i]),
            "n_parents": int(in_degree[i]),
        })

    df = pd.DataFrame(results)
    df = df.sort_values(["level", "n_children"], ascending=[True, False])
    return df


def find_gene_families(score_matrix, gene_names, min_score=0.15):
    """
    找出基因"家族": 一个 root marker 及其所有后代。

    返回: list of dicts, 每个 dict = {parent: str, children: list, grandchildren: list}
    """
    n = len(gene_names)
    gene_names = list(gene_names)

    # 找 root markers (出度高, 入度低)
    hierarchy = classify_hierarchy(score_matrix, gene_names, min_score)
    roots = hierarchy[hierarchy["role"] == "root_marker"]["gene"].tolist()
    intermediates = hierarchy[hierarchy["role"] == "intermediate"]["gene"].tolist()

    families = []
    for root in roots:
        ri = gene_names.index(root)
        # 直接子代
        children = []
        for j in range(n):
            if score_matrix[ri, j] >= min_score:
                children.append(gene_names[j])

        # 孙代 (children 中的 intermediate 的子代)
        grandchildren = []
        for child in children:
            if child in intermediates:
                ci = gene_names.index(child)
                for k in range(n):
                    if score_matrix[ci, k] >= min_score and gene_names[k] not in children and gene_names[k] != root:
                        grandchildren.append(gene_names[k])

        families.append({
            "parent": root,
            "children": children,
            "grandchildren": grandchildren,
            "family_size": 1 + len(children) + len(grandchildren),
        })

    return sorted(families, key=lambda x: x["family_size"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Per Cell-Type-Pair Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_inclusion_per_celltype_pair(X_binary, gene_names, cell_types, min_score=0.1):
    """
    对每对细胞型分别计算包含分数。

    返回: dict of {(ctA, ctB): score_matrix}
    """
    unique_types = sorted(set(cell_types))
    results = {}

    for ctA, ctB in combinations(unique_types, 2):
        # 取这两个细胞型的细胞
        mask = np.isin(cell_types, [ctA, ctB])
        X_sub = X_binary[mask]

        # 计算包含分数
        n_genes = X_sub.shape[1]
        score_mat = np.zeros((n_genes, n_genes))

        for i, j in combinations(range(n_genes), 2):
            si, sj, _, _, _ = compute_directed_score(X_sub[:, i], X_sub[:, j])
            score_mat[i, j] = si
            score_mat[j, i] = sj

        results[(ctA, ctB)] = score_mat

    return results


def find_shared_vs_specific_markers(pair_results, gene_names, min_score=0.15):
    """
    比较不同细胞型对的包含关系，找出:
    - 共有 parent marker (在多对中都出现的 root marker)
    - 特异 child marker (只在特定对中出现的 leaf marker)

    返回: DataFrame with gene, type (shared/specific), pairs
    """
    gene_names = list(gene_names)

    # 统计每个基因在多少对中作为 parent/child 出现
    gene_as_parent = defaultdict(set)  # gene → set of (ctA, ctB)
    gene_as_child = defaultdict(set)

    for (ctA, ctB), score_mat in pair_results.items():
        n = len(gene_names)
        for i in range(n):
            for j in range(n):
                if i != j and score_mat[i, j] >= min_score:
                    gene_as_parent[gene_names[i]].add((ctA, ctB))
                    gene_as_child[gene_names[j]].add((ctA, ctB))

    results = []
    all_pairs = set(pair_results.keys())

    for g in gene_names:
        parent_pairs = gene_as_parent.get(g, set())
        child_pairs = gene_as_child.get(g, set())

        if not parent_pairs and not child_pairs:
            continue

        # 判断是 shared 还是 specific
        if len(parent_pairs) >= len(all_pairs) * 0.5:
            marker_type = "shared_parent"   # 在多数对中都是 parent → 大类 marker
        elif len(child_pairs) >= len(all_pairs) * 0.5:
            marker_type = "shared_child"    # 在多数对中都是 child → 通用亚型
        elif len(parent_pairs) > 0:
            marker_type = "specific_parent" # 只在少数对中是 parent → 特异 marker
        else:
            marker_type = "specific_child"  # 只在少数对中是 child → 特异亚型

        results.append({
            "gene": g,
            "marker_type": marker_type,
            "n_as_parent": len(parent_pairs),
            "n_as_child": len(child_pairs),
            "parent_in_pairs": "; ".join([f"{a}-{b}" for a, b in sorted(parent_pairs)]),
            "child_in_pairs": "; ".join([f"{a}-{b}" for a, b in sorted(child_pairs)]),
        })

    return pd.DataFrame(results).sort_values(
        ["marker_type", "n_as_parent", "n_as_child"], ascending=[True, False, False])


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_inclusion_heatmap(score_matrix, gene_names, top_genes, out_path):
    """包含分数热图 (Row ⊃ Column)"""
    idx = [list(gene_names).index(g) for g in top_genes]
    sub = score_matrix[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sub, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(top_genes)))
    ax.set_xticklabels(top_genes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=8)
    ax.set_xlabel("Included gene (B in A⊃B)", fontsize=12)
    ax.set_ylabel("Parent gene (A in A⊃B)", fontsize=12)
    ax.set_title("Directional Inclusion Score\n(Row ⊃ Column)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)

    for i in range(len(top_genes)):
        for j in range(len(top_genes)):
            if i != j and sub[i, j] > 0.1:
                ax.text(j, i, f"{sub[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if sub[i, j] > 0.5 else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_hierarchy_tree(families, out_path):
    """绘制基因层级树"""
    try:
        import networkx as nx
    except ImportError:
        print("  Skipping hierarchy tree (networkx not installed)")
        return

    G = nx.DiGraph()
    for fam in families[:15]:  # 最多显示 15 个家族
        parent = fam["parent"]
        for child in fam["children"]:
            G.add_edge(parent, child)
        for gc in fam["grandchildren"]:
            # 找 gc 的 parent (在 children 中)
            for child in fam["children"]:
                G.add_edge(child, gc)

    if len(G.nodes()) == 0:
        return

    fig, ax = plt.subplots(figsize=(18, 10))

    # 分层布局: roots 在上, children 在中, grandchildren 在下
    hierarchy = classify_hierarchy(np.zeros((len(G.nodes()), len(G.nodes()))),
                                    list(G.nodes()), min_score=0)
    # 简单分层
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # 节点颜色按层级
    node_colors = []
    for n in G.nodes():
        if G.in_degree(n) == 0:
            node_colors.append("#E63946")   # root: red
        elif G.out_degree(n) > 0:
            node_colors.append("#F4A261")   # intermediate: orange
        else:
            node_colors.append("#2A9D8F")   # leaf: teal

    node_sizes = [500 + 200 * G.out_degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray",
                           arrows=True, arrowsize=12, ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946', markersize=10, label='Root marker (parent)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F4A261', markersize=10, label='Intermediate'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2A9D8F', markersize=10, label='Leaf marker (subtype)'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title("Gene Inclusion Hierarchy\nArrow: A → B means A ⊃ B", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_marker_classification(marker_df, out_path):
    """绘制 marker 分类统计"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: marker 类型分布
    ax = axes[0]
    counts = marker_df["marker_type"].value_counts()
    colors = {"shared_parent": "#E63946", "shared_child": "#F4A261",
              "specific_parent": "#2A9D8F", "specific_child": "#8338EC"}
    bars = ax.barh(counts.index, counts.values,
                   color=[colors.get(c, "#999") for c in counts.index])
    ax.set_xlabel("Number of genes")
    ax.set_title("Marker Classification", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=10)

    # 右: n_as_parent vs n_as_child scatter
    ax = axes[1]
    for mtype in marker_df["marker_type"].unique():
        sub = marker_df[marker_df["marker_type"] == mtype]
        ax.scatter(sub["n_as_parent"], sub["n_as_child"],
                   c=colors.get(mtype, "#999"), label=mtype, s=30, alpha=0.7)
    ax.set_xlabel("Times as parent (includes others)")
    ax.set_ylabel("Times as child (included by others)")
    ax.set_title("Parent vs Child Roles", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_mutex_profile_mode(args):
    """Run inclusion analysis directly from a pre-computed CME matrix."""
    print(f"Loading mutual-exclusion matrix {args.input} ...")
    cme_matrix, gene_names = load_square_matrix(args.input, matrix_key=args.matrix_key)
    print(f"  Genes: {len(gene_names)}")

    if args.gene_list:
        with open(args.gene_list) as f:
            selected = [line.strip() for line in f if line.strip()]
        if len(selected) == cme_matrix.shape[0]:
            gene_names = np.asarray(selected, dtype=object)
            print(f"  Using {len(gene_names)} gene names from --gene-list")
        else:
            name_to_idx = {str(g): idx for idx, g in enumerate(gene_names)}
            selected_idx = [name_to_idx[g] for g in selected if g in name_to_idx]
            if len(selected_idx) < 2:
                raise ValueError("Too few genes from --gene-list match the matrix labels.")
            cme_matrix = cme_matrix[np.ix_(selected_idx, selected_idx)]
            gene_names = gene_names[selected_idx]
            print(f"  Filtered to {len(gene_names)} genes from --gene-list")

    mutex_threshold = args.mutex_threshold
    if mutex_threshold is None:
        mutex_threshold = args.cme_threshold

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.input) or ".",
        "mutex_profile_inclusion_results",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[1/5] Binarizing mutual-exclusion matrix (threshold={mutex_threshold}) ...")
    mutex_binary = binarize_mutex_matrix(cme_matrix, threshold=mutex_threshold)
    degrees = mutex_binary.sum(axis=1)
    print(
        f"  Mutex degree: mean={degrees.mean():.2f}, "
        f"range=[{degrees.min()}, {degrees.max()}]"
    )

    print(f"\n[2/5] Computing relation-profile inclusion scores ...")
    score_matrix, detail_df = compute_mutex_profile_inclusion_matrix(
        mutex_binary,
        gene_names,
        exclude_pair_entries=not args.include_pair_entries,
    )
    print(
        "  Pairs with score > 0: "
        f"{len(detail_df[detail_df[['score_A_sup_B','score_B_sup_A']].max(axis=1) > 0])}"
    )

    print(f"\n[3/5] Finding inclusion pairs (min_score={args.min_score}) ...")
    detail_df["max_score"] = detail_df[["score_A_sup_B", "score_B_sup_A"]].max(axis=1)
    top_pairs = detail_df[detail_df["max_score"] >= args.min_score].sort_values(
        "max_score", ascending=False).head(100)

    if len(top_pairs) == 0:
        print("  WARNING: No pairs found above threshold. Try lowering --min-score or --mutex-threshold")
    else:
        print(f"  Found {len(top_pairs)} inclusion pairs")
        print(f"\n  Top 20:")
        print("  " + "-" * 85)
        for _, row in top_pairs.head(20).iterrows():
            if row["score_A_sup_B"] >= row["score_B_sup_A"]:
                arrow = f"{row['gene_A']} -> {row['gene_B']}"
                sc = row["score_A_sup_B"]
                violation = row["n10"]
                extra = row["n01"]
            else:
                arrow = f"{row['gene_B']} -> {row['gene_A']}"
                sc = row["score_B_sup_A"]
                violation = row["n01"]
                extra = row["n10"]
            print(
                f"  {arrow:<40} score={sc:.4f}  "
                f"shared[1,1]={row['n11']:>4}  violation={violation:>4}  extra={extra:>4}"
            )

    print(f"\n[4/5] Classifying gene hierarchy ...")
    hierarchy = classify_hierarchy(score_matrix, gene_names, args.min_score)
    for role in ["root_marker", "intermediate", "leaf_marker", "independent"]:
        genes = hierarchy[hierarchy["role"] == role]["gene"].tolist()
        if genes:
            print(f"    {role}: {len(genes)} genes")
            if len(genes) <= 10:
                print(f"      {', '.join(genes)}")

    families = find_gene_families(score_matrix, gene_names, args.min_score)
    if families:
        print(f"\n  Gene relation families (top 10):")
        print("  " + "-" * 75)
        for fam in families[:10]:
            children_str = ", ".join(fam["children"][:5])
            if len(fam["children"]) > 5:
                children_str += f" ... (+{len(fam['children'])-5})"
            print(f"  {fam['parent']:<20} -> [{children_str}]  (family_size={fam['family_size']})")

    print(f"\n[5/5] Saving results ...")
    detail_df.drop(columns=["max_score"], errors="ignore").to_csv(
        os.path.join(out_dir, "mutex_profile_inclusion_pairs.csv"),
        index=False,
    )
    top_pairs.to_csv(os.path.join(out_dir, "mutex_profile_inclusion_top_pairs.csv"), index=False)
    pd.DataFrame(score_matrix, index=gene_names, columns=gene_names).to_csv(
        os.path.join(out_dir, "mutex_profile_inclusion_score_matrix.csv"))
    pd.DataFrame(mutex_binary.astype(np.int8), index=gene_names, columns=gene_names).to_csv(
        os.path.join(out_dir, "mutex_binary_matrix.csv"))
    hierarchy.to_csv(os.path.join(out_dir, "mutex_profile_gene_hierarchy.csv"), index=False)

    if families:
        fam_rows = []
        for fam in families:
            fam_rows.append({
                "parent": fam["parent"],
                "children": "; ".join(fam["children"]),
                "grandchildren": "; ".join(fam["grandchildren"]),
                "family_size": fam["family_size"],
            })
        pd.DataFrame(fam_rows).to_csv(
            os.path.join(out_dir, "mutex_profile_gene_families.csv"),
            index=False,
        )

    if len(top_pairs) > 0:
        genes_in_top = set(top_pairs["gene_A"].tolist() + top_pairs["gene_B"].tolist())
        gene_max_scores = {}
        for g in genes_in_top:
            gi = list(gene_names).index(g)
            gene_max_scores[g] = max(score_matrix[gi, :].max(), score_matrix[:, gi].max())
        top_genes = sorted(gene_max_scores, key=gene_max_scores.get, reverse=True)[:args.top]
        plot_inclusion_heatmap(
            score_matrix,
            gene_names,
            top_genes,
            os.path.join(out_dir, "mutex_profile_inclusion_heatmap.png"),
        )
        if families:
            plot_hierarchy_tree(
                families,
                os.path.join(out_dir, "mutex_profile_inclusion_hierarchy.png"),
            )

    print(f"\n✓ Done! Results in: {out_dir}")
    print(f"  Files:")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"    {f} ({size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Directional Inclusion Score")
    parser.add_argument("input", help="Input .h5ad file, or CME matrix with --mutex-profile")
    parser.add_argument("--mutex-profile", action="store_true",
                        help="Treat input as a pre-computed CME/mutual-exclusion score matrix")
    parser.add_argument("--mutex-threshold", type=float, default=None,
                        help="Threshold for binarizing CME matrix in --mutex-profile mode")
    parser.add_argument("--matrix-key", default="cme_matrix",
                        help="NPZ key for matrix input in --mutex-profile mode (default: cme_matrix)")
    parser.add_argument("--include-pair-entries", action="store_true",
                        help="Include the queried gene-pair entries when comparing mutex profiles")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Binarization threshold (default: 1.0)")
    parser.add_argument("--min-score", type=float, default=0.15,
                        help="Minimum score to report (default: 0.15)")
    parser.add_argument("--top", type=int, default=30,
                        help="Top genes for heatmap (default: 30)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--gene-list", default=None,
                        help="File with gene names to analyze (one per line)")
    parser.add_argument("--cme-matrix", default=None,
                        help="Pre-computed CME matrix (CSV) for pre-filtering")
    parser.add_argument("--cme-threshold", type=float, default=0.5,
                        help="CME threshold for pre-filtering (default: 0.5)")
    parser.add_argument("--by-celltype", action="store_true",
                        help="Analyze per cell-type pair (requires cell_type in obs)")
    args = parser.parse_args()

    if args.mutex_profile:
        run_mutex_profile_mode(args)
        return

    # ── Load data ──
    import scanpy as sc
    print(f"Loading {args.input} ...")
    adata = sc.read_h5ad(args.input)
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float64)

    # Gene list filtering
    if args.gene_list:
        with open(args.gene_list) as f:
            selected = [line.strip() for line in f if line.strip()]
        mask = [g in selected for g in adata.var_names]
        X = X[:, mask]
        gene_names = np.array(adata.var_names)[mask]
        print(f"  Filtered to {len(gene_names)} genes")
    else:
        gene_names = np.array(adata.var_names)

    # CME pre-filtering
    if args.cme_matrix:
        print(f"\n  Loading CME matrix: {args.cme_matrix}")
        cme = pd.read_csv(args.cme_matrix, index_col=0)
        cme_genes = [g for g in gene_names if g in cme.index]
        if len(cme_genes) < 2:
            print("  ERROR: Too few genes match between CME matrix and expression data")
            sys.exit(1)

        # Find gene pairs with high CME (mutual exclusion)
        high_cme_pairs = set()
        for i, j in combinations(range(len(cme_genes)), 2):
            g1, g2 = cme_genes[i], cme_genes[j]
            if cme.loc[g1, g2] >= args.cme_threshold:
                high_cme_pairs.add((g1, g2))
                high_cme_pairs.add((g2, g1))

        print(f"  Gene pairs with CME ≥ {args.cme_threshold}: {len(high_cme_pairs) // 2}")

        # Filter X to CME genes
        mask = [g in cme_genes for g in gene_names]
        X = X[:, mask]
        gene_names = np.array(gene_names)[mask]
        print(f"  Using {len(gene_names)} genes from CME matrix")
    else:
        high_cme_pairs = None

    # Output directory
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input) or ".", "inclusion_results")
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Binarize ──
    print(f"\n[1/5] Binarizing (threshold={args.threshold}) ...")
    X_bin = binarize_matrix(X, threshold=args.threshold)
    active_pct = X_bin.sum(axis=0) / X_bin.shape[0]
    print(f"  Active rate: mean={active_pct.mean():.2%}, range=[{active_pct.min():.2%}, {active_pct.max():.2%}]")

    # ── Step 2: Compute inclusion scores ──
    print(f"\n[2/5] Computing inclusion scores ({len(gene_names)} genes) ...")
    score_matrix, detail_df = compute_inclusion_matrix(X_bin, gene_names)

    # Filter by CME if provided
    if high_cme_pairs is not None:
        before = len(detail_df)
        detail_df = detail_df[
            detail_df.apply(lambda r: (r["gene_A"], r["gene_B"]) in high_cme_pairs, axis=1)
        ]
        print(f"  Filtered by CME: {before} → {len(detail_df)} pairs")

    print(f"  Pairs with score > 0: {len(detail_df[detail_df[['score_A_sup_B','score_B_sup_A']].max(axis=1) > 0])}")

    # ── Step 3: Find top pairs ──
    print(f"\n[3/5] Finding inclusion pairs (min_score={args.min_score}) ...")
    detail_df["max_score"] = detail_df[["score_A_sup_B", "score_B_sup_A"]].max(axis=1)
    top_pairs = detail_df[detail_df["max_score"] >= args.min_score].sort_values(
        "max_score", ascending=False).head(100)

    if len(top_pairs) == 0:
        print("  WARNING: No pairs found above threshold. Try lowering --min-score")
    else:
        print(f"  Found {len(top_pairs)} inclusion pairs")
        print(f"\n  Top 20:")
        print("  " + "-" * 75)
        for _, row in top_pairs.head(20).iterrows():
            if row["score_A_sup_B"] >= row["score_B_sup_A"]:
                arrow = f"{row['gene_A']} ⊃ {row['gene_B']}"
                sc = row["score_A_sup_B"]
            else:
                arrow = f"{row['gene_B']} ⊃ {row['gene_A']}"
                sc = row["score_B_sup_A"]
            print(f"  {arrow:<40} score={sc:.4f}  [1,1]={row['n11']:>4}  [1,0]={row['n10']:>4}  [0,1]={row['n01']:>4}")

    # ── Step 4: Hierarchy classification ──
    print(f"\n[4/5] Classifying gene hierarchy ...")
    hierarchy = classify_hierarchy(score_matrix, gene_names, args.min_score)

    print(f"\n  Gene roles:")
    for role in ["root_marker", "intermediate", "leaf_marker", "independent"]:
        genes = hierarchy[hierarchy["role"] == role]["gene"].tolist()
        if genes:
            print(f"    {role}: {len(genes)} genes")
            if len(genes) <= 10:
                print(f"      {', '.join(genes)}")

    # Gene families
    families = find_gene_families(score_matrix, gene_names, args.min_score)
    if families:
        print(f"\n  Gene families (top 10):")
        print("  " + "-" * 75)
        for fam in families[:10]:
            children_str = ", ".join(fam["children"][:5])
            if len(fam["children"]) > 5:
                children_str += f" ... (+{len(fam['children'])-5})"
            print(f"  {fam['parent']:<20} → [{children_str}]  (family_size={fam['family_size']})")

    # ── Step 5: Per cell-type-pair analysis ──
    if args.by_celltype and "cell_type" in adata.obs.columns:
        print(f"\n[5/5] Per cell-type-pair analysis ...")
        cell_types = adata.obs["cell_type"].values
        pair_results = compute_inclusion_per_celltype_pair(X_bin, gene_names, cell_types, args.min_score)

        marker_df = find_shared_vs_specific_markers(pair_results, gene_names, args.min_score)
        marker_path = os.path.join(out_dir, "marker_classification.csv")
        marker_df.to_csv(marker_path, index=False)
        print(f"  Marker classification saved to: {marker_path}")

        # Summary
        for mtype in ["shared_parent", "specific_parent", "shared_child", "specific_child"]:
            genes = marker_df[marker_df["marker_type"] == mtype]["gene"].tolist()
            if genes:
                print(f"\n    {mtype} ({len(genes)} genes):")
                print(f"      {', '.join(genes[:15])}" + (" ..." if len(genes) > 15 else ""))

        # Plot marker classification
        plot_marker_classification(marker_df, os.path.join(out_dir, "marker_classification.png"))
    else:
        print(f"\n[5/5] Skipping per-celltype analysis (use --by-celltype to enable)")

    # ── Save results ──
    print(f"\n[Save] Saving results ...")

    # All pairs
    detail_df = detail_df.drop(columns=["max_score"], errors="ignore")
    detail_df.to_csv(os.path.join(out_dir, "inclusion_pairs.csv"), index=False)

    # Top pairs
    top_pairs.to_csv(os.path.join(out_dir, "inclusion_top_pairs.csv"), index=False)

    # Score matrix
    pd.DataFrame(score_matrix, index=gene_names, columns=gene_names).to_csv(
        os.path.join(out_dir, "inclusion_score_matrix.csv"))

    # Hierarchy
    hierarchy.to_csv(os.path.join(out_dir, "gene_hierarchy.csv"), index=False)

    # Families
    if families:
        fam_rows = []
        for fam in families:
            fam_rows.append({
                "parent": fam["parent"],
                "children": "; ".join(fam["children"]),
                "grandchildren": "; ".join(fam["grandchildren"]),
                "family_size": fam["family_size"],
            })
        pd.DataFrame(fam_rows).to_csv(os.path.join(out_dir, "gene_families.csv"), index=False)

    # ── Visualizations ──
    print(f"\n[Viz] Generating plots ...")

    if len(top_pairs) > 0:
        # Heatmap
        genes_in_top = set(top_pairs["gene_A"].tolist() + top_pairs["gene_B"].tolist())
        gene_max_scores = {}
        for g in genes_in_top:
            gi = list(gene_names).index(g)
            gene_max_scores[g] = max(score_matrix[gi, :].max(), score_matrix[:, gi].max())
        top_genes = sorted(gene_max_scores, key=gene_max_scores.get, reverse=True)[:args.top]

        plot_inclusion_heatmap(score_matrix, gene_names, top_genes,
                               os.path.join(out_dir, "inclusion_heatmap.png"))
        print("  inclusion_heatmap.png")

        # Hierarchy tree
        if families:
            plot_hierarchy_tree(families, os.path.join(out_dir, "inclusion_hierarchy.png"))
            print("  inclusion_hierarchy.png")

    print(f"\n✓ Done! Results in: {out_dir}")
    print(f"  Files:")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"    {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
