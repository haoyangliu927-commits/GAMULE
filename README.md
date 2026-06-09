# Clean Hyperedge Package

这个文件夹包含从 scRNA-seq expression 到 CME/Jaccard 正负监督，再到超边 gene module 推断的最小本地文件：

- `clean_supervision_run.ipynb`
- `CME_CPU.py`
- `src/supervision_pipeline.py`
- `src/cme_supervision.py`
- `src/cme_visualization.py`
- `src/train.py`
- `src/loss.py`
- `scripts/test_cme_supervision.py`
- `docs/cme_to_gene_modules.html`
- `datasets/adata_672.h5ad`

在 Jupyter 里把工作目录设为这个文件夹，打开并运行 `clean_supervision_run.ipynb` 即可。环境里仍然需要已安装 `torch`、`scanpy`、`matplotlib` 等 Python 依赖。

命令行测试：

```bash
/home/luqi/miniforge3/envs/gamule/bin/python clean_HYperedge/scripts/test_cme_supervision.py
```

测试会生成：

- `results/cme_supervision_heatmaps.png`
- `results/cme_supervision_matrices.npz`
- `results/cme_supervision_stats.json`

当前默认设计是 `6 + 1` 个超边：

- 前 6 个是有意义的 gene modules，参与训练和相似度计算。
- 最后 1 个是无意义/未分配超边，只接收完全没有出现在任何监督矩阵里的基因。
- 第 7 个超边不参与训练时的 `partition @ hyperedge_emb`，最终 `gene_emb` 也只用前 6 个超边生成，因此不会影响有意义 gene modules 的相似度。
