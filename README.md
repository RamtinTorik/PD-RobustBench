# Robustness Framework for Deep-Learning Parkinson's Disease Detection from Drawings

`PD-RobustBench` — a reproducible evaluation framework that measures how much
deep-learning models for **Parkinson's disease (PD) detection from
handwriting/drawing images** degrade under realistic image corruptions, and how
their **Grad-CAM explanations** change under the same corruptions.

```
Parkinson's Drawings (spiral, wave)  +  HandPD (spiral, meander)
        │
        ▼
3 pretrained CNNs (ResNet50 / VGG16 / EfficientNet-B0, frozen backbone + light head)
        │
        ▼
6 corruption families × 5 severity levels (ImageNet-C style, applied on the fly)
        │
        ├──► metrics: accuracy, balanced accuracy, precision, sensitivity,
        │             specificity, F1, ROC-AUC  (clean + every condition)
        ├──► Grad-CAM stability: Spearman/Pearson/SSIM/IoU(top-20%) vs clean CAM
        └──► Parkinson Robustness Score (PRS, 0–100, higher = more robust)
```

## Repository layout

```
article_project/
├── src/
│   ├── config.py               # paths, seeds, corruption levels, PRS weights
│   ├── corruptions.py          # the 6 corruption families (task 3.2)
│   ├── data.py                 # dataset preparation/loading (PDRAW + HandPD)
│   ├── models.py               # ResNet50 / VGG16 / EfficientNet-B0 + light head
│   ├── train.py                # head training on cached frozen features (task 4.1)
│   ├── inference.py            # prediction helpers + clean baseline (tasks 4.1/4.2)
│   ├── metrics.py              # accuracy/sensitivity/specificity/F1/AUC (task 4.2)
│   ├── run_robustness.py       # main loop: model × corruption × level (task 4.3)
│   ├── merge_results.py        # merges per-process part CSVs + integrity check
│   ├── gradcam.py              # hand-written Grad-CAM (task 5.1)
│   ├── run_xai.py              # XAI subset + stability metrics (tasks 5.2/5.3)
│   ├── robustness_score.py     # Parkinson Robustness Score, clipped to [0,100]
│   ├── prs_sensitivity.py      # PRS weight-sensitivity ablation (review fix)
│   ├── gradcam_sanity.py       # model-randomization control for Grad-CAM
│   ├── supplementary_stats.py  # bootstrap CIs, parameter counts, thresholds
│   ├── make_corruption_config.py  # writes docs/task3_1_corruption_levels.csv
│   └── make_figures.py         # all paper figures/tables (task 6.2)
├── docs/
│   ├── task1_1_models_datasets.md / .csv     # literature: models-datasets-accuracy
│   ├── task1_2_related_work_summary.md       # literature: robustness frameworks
│   ├── task3_1_corruption_levels.csv         # corruption benchmark specification
│   └── paper/
│       ├── main.tex                          # the manuscript
│       └── submission/                       # self-contained Overleaf bundle
│           ├── main.tex, supplementary.tex   # (+ figures/, table row files)
│           └── ../paper_submission.zip       # upload this to Overleaf
├── data/raw/<dataset>_<dtype>/{train,test}/{healthy,parkinson}/
├── results/                    # CSVs: robustness_results, predictions, scores
│   ├── figures/                # all generated figures (+ corrupted_samples/)
│   └── tables/                 # final paper tables (CSV + LaTeX)
├── xai_outputs/                # saved Grad-CAM maps (.npy)
├── checkpoints/                # trained heads + training_summary.csv
└── requirements.txt
```

## Setup

```bash
python -m pip install -r requirements.txt
```

- Python 3.10+, PyTorch (CPU build is sufficient — the whole benchmark runs in
  well under an hour on a modern laptop CPU), torchvision, OpenCV, scikit-learn,
  scikit-image, scipy, pandas, matplotlib, seaborn.

## Datasets

| Key | Source | Content | Split |
|-----|--------|---------|-------|
| `pdraw_spiral`, `pdraw_wave` | [Kaggle: parkinsons-drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings) (Zham et al. data) | 256×256 PNG | official: 72 train / 30 test per type |
| `handpd_spiral`, `handpd_meander` | [HandPD, UNESP](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/) (Pereira et al. 2016) | ~700×700 JPG, 92 subjects | our subject-wise 70/30 split (13/5 healthy, 52/22 PD subjects) |

Both are fetched automatically: the Kaggle one via `kagglehub`
(`kagglehub.dataset_download("kmader/parkinsons-drawings")`), HandPD by placing
`Spiral_HandPD.zip`/`Meander_HandPD.zip` from the UNESP page into `data/` and
extracting to `data/handpd_tmp/`. `python -c "from data import prepare_all"` (run
inside `src/`) then copies everything into the canonical folder layout.

## Run the full benchmark

```bash
cd src
python train.py                # adapts 16 models (4 data types × 4 models):
                               # stage 1 = light head on frozen features,
                               # stage 2 = ≤10-epoch fine-tune of the last
                               # backbone block only (backbones stay ImageNet)
python run_robustness.py       # clean + 8×5 corrupted evaluations -> results/
python run_xai.py --dtype pdraw_spiral    # Grad-CAM subset (repeat per dtype)
python robustness_score.py     # PRS per model (clipped to [0,100])
python make_figures.py         # final figures + LaTeX tables
```

For faster wall-clock execution the same commands accept `--dtype` and
`--out`, so training/evaluation can run as four parallel processes (one per
data type); `python merge_results.py --parts <dir> --tag <tag>` then
assembles the canonical CSVs. Multi-seed runs add `--seed 42|43|44` to
train/run_robustness/run_xai (checkpoints for non-default seeds are suffixed
`__s<seed>`), each seed's parts live in `results/parts/s<seed>/`, and
`python robustness_score.py --results ... --xai ... --tag _s<seed>` +
`python aggregate_seeds.py` produce the cross-seed mean ± std
(`results/seed_aggregate.csv`).

## Corruption benchmark (task 3.1)

| Family | Parameter | Severity 1 → 5 |
|--------|-----------|----------------|
| jpeg_compression | JPEG quality | 90, 70, 50, 30, 10 |
| gaussian_blur | kernel size (px) | 3, 5, 7, 9, 11 |
| gaussian_noise | σ (fraction of 255) | 0.01, 0.03, 0.05, 0.08, 0.12 |
| contrast | factor (mean-centred) | 0.50, 0.70, 0.85, 1.30, 1.50 |
| brightness | factor | 0.50, 0.70, 0.85, 1.30, 1.50 |
| shadow | brightness at darkest corner of a diagonal gradient | 0.85, 0.70, 0.55, 0.42, 0.30 |
| perspective | corner displacement (fraction of min(h,w)) | 0.02, 0.04, 0.06, 0.08, 0.10 |
| rotation | degrees (alternating direction) | −5, 10, −15, 20, −25 |

Corruptions are **applied on the fly** during inference (deterministic RNG
seeding per corruption+level), and a visual gallery of one image under all
30 conditions is saved to `results/figures/corrupted_samples/`.

## Parkinson Robustness Score (task 6.1)

For each model *m* and dataset type, per corruption family *c*:

```
ΔAcc_c = mean_level (clean_bAcc − bAcc) / clean_bAcc      # relative drop
ΔF1_c  = mean_level (clean_F1  − F1)  / clean_F1
ΔXAI_c = mean_level (1 − Spearman(cam_corrupted, cam_clean))
D_c    = 0.5·ΔAcc_c + 0.3·ΔF1_c + 0.2·ΔXAI_c
PRS_m  = 100 · (1 − mean_c D_c)                            # 0–100, higher = better
```

Balanced accuracy is used in the accuracy term because HandPD is ~4:1 imbalanced
(for the balanced PDRAW sets it coincides with plain accuracy); a plain-accuracy
variant is also reported.

## Reproducibility

- Global seed 42 (`src/config.py`); deterministic corruption RNG; fixed
  subject-wise HandPD split; fixed XAI subset (5 PD + 5 healthy per type).
- Exact library versions in `requirements.txt`.

## License / data credits

Code: MIT. Datasets: Parkinson's Drawings (Zham et al. 2016/2017; Kaggle release
by K. Scott Mader) and HandPD (Pereira et al. 2016, UNESP) — cite the original
papers when using this framework.
