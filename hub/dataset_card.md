---
license: cc-by-4.0
tags:
  - spatial-transcriptomics
  - merfish
  - mouse-brain
  - single-cell
pretty_name: MIMYR benchmark data
---

# MIMYR benchmark data

The preprocessed tissue sections used to train and evaluate
[MIMYR](https://github.com/gkrieg/mimyr), a generative model of missing tissue in
spatial transcriptomics. Each section is an AnnData `.h5ad` file whose cells have
CCF coordinates (`x_ccf`, `y_ccf`, `z_ccf`), Allen Brain Cell Atlas cell-type labels
(`class`, `subclass`, `supertype`, `cluster`), and a gene expression matrix.

MIMYR downloads only the archives a run needs, and extracts them into its
`--data_dir`:

| Archive | Contents | MIMYR data modes |
|---|---|---|
| `rq1-merscope.zip` | 53 coronal sections of one mouse brain (donor C57BL6J-638850), 550 genes | `rq1` |
| `Zhuang-ABCA-1.zip` | Zhuang-ABCA-1 (147 coronal sections), 1,122 genes, plus `ccf-ABCA-1.csv` | `base_zhuang1`, `rq4_noref` |
| `Zhuang-ABCA-2.zip` | Zhuang-ABCA-2 (66 coronal sections), plus `ccf-ABCA-2.csv` | `rq3`, `rq3_v2*`, `rq4_rq3` |
| `Zhuang-ABCA-3.zip` | Zhuang-ABCA-3 (23 sagittal sections), plus `ccf-ABCA-3.csv` | `rq4*` |
| `Zhuang-ABCA-4.zip` | Zhuang-ABCA-4 (3 sections), plus `ccf-ABCA-4.csv` | — |

The files are identical to the ones used for the paper.

## Sources and attribution

All sections are derived from data released through the
[Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas)
under CC BY 4.0. If you use them, please cite the original datasets as well as MIMYR:

- **MERSCOPE whole-brain atlas** (`rq1-merscope.zip`): Yao Z, et al. A high-resolution
  transcriptomic and spatial atlas of cell types in the whole mouse brain.
  *Nature* 624, 317–332 (2023).
- **Zhuang-ABCA-1 to 4**: Zhang M, et al. Molecularly defined and spatially resolved
  cell atlas of the whole mouse brain. *Nature* 624, 343–354 (2023).

## Citation

> Deshpande A, Bei Z, Ma J, Krieger S. MIMYR: Generative modeling of missing tissue
> in spatial transcriptomics. *bioRxiv* [Preprint]. 2025 Nov 27:2025.11.24.690239.
> doi: [10.1101/2025.11.24.690239](https://doi.org/10.1101/2025.11.24.690239).
> PMID: 41394599; PMCID: PMC12697367.
