# Reproducing the manuscript

Lesion-necessity testing on the Cheng et al. brain tumor MRI collection, with a
HAM10000 positive control. Every number in the manuscript comes from one of the
scripts below. Training is deterministic for a fixed seed (`core/seeding.py`); a
repeated run of the occluded SmallCNN condition (`a6l`) reproduced the per-fold
macro-F1 of Table 1 to four decimal places.

## Environment
Kaggle, one NVIDIA T4. torch 2.10, captum 0.9, scikit-learn, scipy, h5py.
Run every script from the repository root with `PYTHONPATH=.`:

    python -c "from core import data_mat; data_mat.build_cache()"   # brain cache, ~4 min
    PYTHONPATH=. python scripts/<script>.py

## Data (public, not redistributed here)
| Collection | Source | Used for |
|---|---|---|
| Cheng et al. brain tumor (.mat, with masks) | figshare, DOI 10.6084/m9.figshare.1512427 | Sections 3.1-3.5, 3.8 |
| Four-class aggregate (no-tumor class) | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` | Section 3.6 |
| HAM10000 images and lesion masks | Tschandl et al. 2018 / 2020 | Sections 3.7, 3.9 |

## Script -> manuscript map
| Manuscript item | Script | Saved output |
|---|---|---|
| Section 3.1 baselines | `a3_centralized.py`, `a3b_freeze.py` | console only |
| Tables 1-2, Fig. 2, variance decomposition | `a6e_resnet_replication.py` | `results/brain/a6e_results.json` |
| Section 3.2 footprint controls (mask-only) | `a6l_mask_shape_leak.py` | `results/brain/a6l_console.log`, `a6l_results.json` |
| Table 3, Fig. 3 (resolution) | `a6j_resolution_3seed.py` | console only |
| Table 4, Section 3.4 geometry surrogates | `a6g_subject_vs_class.py`, `a6l_mask_shape_leak.py` (Part A) | `results/brain/a6l_console.log` |
| Section 3.5 class signal vs patient identity | `a6g_subject_vs_class.py` | console only |
| Table 5, Section 3.6 (no-tumor class) | `a6h_aggregate_geometry.py`, `a6i_notumor_control.py` | `results/brain/a6i_console.log` |
| Section 2.1.2 mask-size control and matching | `h0_assemble.py`, `h0b_areamatch.py` | `results/ham/h0*_console.log`, `ham_matched_ids.npy` |
| HAM recipe calibration (Section 2.3) | `h1a_baseline.py` | `results/ham/h1a_console.log` |
| Table 6 (HAM necessity) | `h1_necessity_v2.py` (adopted), `h1_necessity.py` (brain recipe, Section 3.7) | `results/ham/h1_results_v2.json`, `h1_results_v1_underfit.json` |
| Table 7, Fig. 6 (brain attribution) | `a6k_sanity_check.py`, `make_fig345_jiim.py` (`figure5`) | console only |
| Table 8 (HAM attribution) | `h2_attribution.py`, `h2b_centreprior.py` | `results/ham/h2*.json` |
| Fig. 1 (protocol schematic) | `h4_protocol_figure.py` | - |
| Fig. 2 forest plot / Fig. 3 resolution | `make_fig345_jiim.py` (`figure3`, `figure4`) | - |
| Fig. 4 occluded images / Fig. 5 geometry | `make_fig12_jiim.py` (`figure1`, `figure2`) | - |

Figure-function names inside the scripts follow an earlier numbering; the column
above gives the figure number used in the manuscript.

`a6h_aggregate_geometry.py` reads a patient map of the four-class aggregate
(`--pmap`); it is not redistributed here and is available from Bassi, Zenodo,
DOI 10.5281/zenodo.22755070. The manuscript's Table 5 values come from
`a6i_notumor_control.py`, which does not need it.
