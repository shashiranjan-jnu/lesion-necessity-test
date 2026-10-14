# A Lesion-Necessity Test for Public Imaging Benchmarks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23360008.svg)](https://doi.org/10.5281/zenodo.23360008)

Code and saved outputs for the manuscript

> **A Lesion-Necessity Test for Public Imaging Benchmarks: Brain Tumor MRI
> Classification Without the Tumor.** Shashi Ranjan, Poonguzhali N.
> Submitted to the *Journal of Imaging Informatics in Medicine*.

The test retrains a classifier on images in which the annotated lesion is deleted
(necessity) or isolated (sufficiency), under identical subject-grouped folds and
seeds, and judges each paired contrast with a patient-clustered bootstrap interval.
It is applied to the Cheng et al. brain tumor MRI collection and, as a positive
control, to a lesion-size-matched subset of HAM10000.

## Repository layout

```
core/       data loaders, models, attribution methods, seeding, figure style
scripts/    one script per analysis (see "Script -> manuscript map" below)
results/    saved outputs of the main runs (JSON and console logs)
REPRODUCE.md  detailed script-to-table map
```

## Data (not redistributed)

| Collection | Source | Licence |
|---|---|---|
| Cheng et al. brain tumor collection (`.mat`, with tumor masks) | figshare, DOI 10.6084/m9.figshare.1512427 | CC BY 4.0 |
| Four-class aggregate (no-tumor class only, Sec. 3.6) | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` | see Kaggle page |
| HAM10000 images and lesion masks | Tschandl et al., Sci Data 2018 (10.1038/sdata.2018.161); Nat Med 2020 (10.1038/s41591-020-0942-0) | CC BY-NC |

`results/ham/ham_matched_ids.npy` lists only HAM10000 lesion identifiers of the
size-matched subset, no image data.

## Running on Kaggle (the environment used for every result)

Settings: GPU T4, Internet on. Attach the Cheng `.mat` dataset (and HAM10000 for
the `h*` scripts). If your Kaggle dataset path differs, set `ROOT_DEFAULT` in
`core/data_mat.py` (brain) or check the glob in `core/data_ham.py` (HAM10000).

```python
# Cell 1 - environment
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
!pip install -q opacus captum h5py scipy

# Cell 2 - get the code (public repository, no token needed)
!rm -rf /tmp/code && git clone --depth 1 https://github.com/shashiranjan-jnu/lesion-necessity-test.git /tmp/code
import sys; sys.path.insert(0, "/tmp/code")

# Cell 3 - build the brain cache (~4 min; /kaggle/working is wiped each session)
from core import data_mat; data_mat.build_cache()

# Cell 4 - run any analysis from the repository root
!cd /tmp/code && PYTHONPATH=/tmp/code python scripts/a6e_resnet_replication.py
```

`PYTHONPATH` must point to the repository root. Restart the kernel after editing a
file in `core/`. Outputs are written to `/kaggle/working/`.

## Script -> manuscript map

| Manuscript item | Script |
|---|---|
| Sec. 3.1 baselines | `a3_centralized.py`, `a3b_freeze.py` |
| Tables 1-2, Fig. 2 | `a6e_resnet_replication.py` |
| Sec. 3.2 footprint controls; Sec. 3.4 intact vs occluded geometry | `a6l_mask_shape_leak.py` |
| Table 3, Fig. 3 | `a6j_resolution_3seed.py` (uses `a6d_shortcut_rigour.py`) |
| Table 4, Secs. 3.4-3.5 | `a6g_subject_vs_class.py` |
| Table 5, Sec. 3.6 | `a6h_aggregate_geometry.py`, `a6i_notumor_control.py` |
| Sec. 2.1.2 mask-size control and matching | `h0_assemble.py`, `h0b_areamatch.py` |
| Sec. 2.3 HAM recipe calibration | `h1a_baseline.py` |
| Table 6 | `h1_necessity_v2.py`; `h1_necessity.py` (brain recipe, Sec. 3.7) |
| Table 7, Fig. 6 | `a6k_sanity_check.py`, `make_fig345_jiim.py` |
| Table 8 | `h2_attribution.py`, `h2b_centreprior.py` |
| Figs. 1-5 | `h4_protocol_figure.py`, `make_fig345_jiim.py`, `make_fig12_jiim.py` |

`REPRODUCE.md` gives the saved output for each item. Training is deterministic for a
fixed seed; a repeated run of the occluded SmallCNN condition (`a6l`) reproduced the
per-fold macro-F1 values of Table 1 to four decimal places.

## Citation

Code archive: Zenodo, DOI [10.5281/zenodo.23360008](https://doi.org/10.5281/zenodo.23360008)
(all versions). See `CITATION.cff`. Please also cite the data sources listed above.

## Licence

Code: MIT (see `LICENSE`). Data are subject to the licences of their original
providers.
