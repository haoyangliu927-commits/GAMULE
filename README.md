# GAMULE

GAMULE is a small workflow for discovering hierarchical gene modules from single-cell RNA-seq data.

The current version builds gene-gene relation supervision from CME / p-value / inclusion scores, trains hyperedge-based gene modules, and then infers module-level hierarchy from directed inclusion relationships.

## Main Files

- `run.py`: default example script.
- `run_306.py`: simulation 306 example with 6 biological gene modules and 1 garbage hyperedge.
- `src/`: core implementation.
- `datasets/adata_306.h5ad`: example simulated scRNA-seq data.
- `trees/tree_306.xml`: reference hierarchy for the 306 simulation.

## Install

Use Python 3.11 if possible.

```bash
pip install -r requirements.txt
```

The main dependencies are `torch`, `scanpy`, `numpy`, `pandas`, `matplotlib`, `seaborn`, and `numba`.

## Run Example

```bash
python run_306.py
```

The script will create:

```text
results/adata_306_results/
```

Key outputs include:

- `summary.json`
- `gene_modules.csv`
- `gene_assignment_diagnostics.csv`
- `garbage_genes_only.csv`
- `training_loss_history.png`
- `combined_supervision_heatmaps.png`
- `module_inclusion_hierarchy.png`

`results/` is ignored by git, so results are generated locally after running the script.

## Method Overview

1. Compute CME and p-value matrices from the expression matrix.
2. Build three supervision masks:
   - positive gene pairs
   - negative / mutually exclusive gene pairs
   - weak positive pairs from inclusion relationships
3. Train gene-to-hyperedge soft assignments.
4. Use a garbage hyperedge for broad or weakly informative genes.
5. Aggregate gene-level directed inclusion into module-level hierarchy.

## Notes

If a matching XML file exists in `trees/`, the script uses it to evaluate cell-type and hierarchy accuracy for simulated data. For real data without XML, this evaluation step is skipped.
