# Task 1.1 — Models & Datasets for Image-Based Parkinson's Disease Detection

Literature compilation for the robustness-framework paper. Every row was
checked against a real source (paper page / arXiv / journal site / repository)
on 2026-09-08. Numbers marked *(as cited in Huang et al. 2024)* were taken from
the related-work tables of Huang et al., Information 2024, 15(4):220, not from
the original papers.

## A. Model–Dataset–Accuracy table

| # | Model | Dataset | Data type | Reported accuracy | Paper (authors, year, venue) | Link |
|---|-------|---------|-----------|-------------------|------------------------------|------|
| 1 | VGG16 (no augmentation) | Parkinson's Drawings (NIATS/Uberlândia release; 72 train / 30 test imgs per type) | wave | 92.00% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 2 | VGG16 (no augmentation) | Parkinson's Drawings | spiral | 77.34% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 3 | VGG19 (rotation+flip aug.) | Parkinson's Drawings | wave | 96.67% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 4 | VGG19 (cosine annealing) | Parkinson's Drawings | spiral | 87.66% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 5 | ResNet18 (mixup) | Parkinson's Drawings | wave | 93.00% (test), MCC 0.94 | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 6 | ResNet50 (no augmentation) | Parkinson's Drawings | wave | 94.34% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 7 | ResNet50 (no augmentation) | Parkinson's Drawings | spiral | 89.14% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 8 | ResNet101 (mixup) | Parkinson's Drawings | wave | 96.67% (test), MCC 0.94 | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 9 | ViT (no augmentation) | Parkinson's Drawings | spiral | 93.33% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 10 | ViT (no augmentation) | Parkinson's Drawings | wave | 96.67% (test) | Huang et al., 2024, *Information* 15(4):220 | https://www.mdpi.com/2078-2489/15/4/220 |
| 11 | Improved VGG-19 (deep transfer learning) | Parkinson's Drawings (spiral) | spiral | 91.36% (test) | Frontiers in Medicine, 2024, art. 1453743 | https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1453743/full |
| 12 | VGG16 (+CNN pipeline) | Parkinson's Drawings | spiral + wave | ResNet50 87% on wave images (comparative table) | AIMS Mathematics, 2024, 9(10):28012–28037 | https://www.aimspress.com/article/doi/10.3934/math.2024334 |
| 13 | Deep-learning framework (transfer learning) | Parkinson's Drawings (public Kaggle release) | spiral/wave | ≈87% (abstract; unverified detail) | Razaq et al., 2025, *Diagnostics* 15(21):2795 | https://www.mdpi.com/2075-4418/15/21/2795 |
| 14 | MobileNet-based pipeline | spiral drawings (800 images incl. PD Drawings) | spiral | 91.36% *(as cited in Huang et al. 2024)* | Basnin et al., 2021 (Comput. Intell. Neurosci.) | via https://www.mdpi.com/2078-2489/15/4/220 |
| 15 | CNN (custom) | spiral/wave (10-fold) | spiral, wave | 89% spiral / 88% wave *(as cited in Huang et al. 2024)* | Shaban et al., 2020 | via https://www.mdpi.com/2078-2489/15/4/220 |
| 16 | SVM + LogReg (feature-based, patient-level) | HandPD + NewHandPD (combined) | spiral, meander | 94.44% patient-level | ML4H 2021 workshop paper (arXiv:2111.14781) | https://arxiv.org/abs/2111.14781 |
| 17 | SVM on kinematic/dynamic features | PaHaW-style online handwriting | various tasks | 85.61% (SVM, in-air movement) | Drotár et al., 2014, *Computer Methods and Programs in Biomedicine* 117(3):405–411 | https://doi.org/10.1016/j.cmpb.2014.09.005 *(unverified DOI)* |
| 18 | Guided-spiral kinematic classifier (14 speed/pressure features) | digitizing-tablet spirals (62 subjects; origin of the Kaggle PD-Drawings images) | guided spiral | classification reported, exact value in paper (unverified) | Zham et al., 2016, IEEE EMBC, pp. 5007–5010 | https://doi.org/10.1109/EMBC.2016.7791670 *(unverified DOI)* |
| 19 | OPF/SVM on texture descriptors | HandPD (micrograph scans) | spiral, meander | (numbers in paper; unverified here) | Pereira et al., 2016, *Computer Methods and Programs in Biomedicine* 136:79–88 | https://www.sciencedirect.com/science/article/pii/S0169260716301894 |

Community code implementing PD spiral/wave classification (used to sanity-check
model choices; no peer-reviewed numbers):

- https://github.com/Yuvnish017/Parkinsons_Disease_Detection_using_Parkinsons_Spiral_Drawing (Kaggle PD Drawings, CNN)
- https://github.com/Sagnik2003/Parkinson-s-Detection-using-HandPD-data (HandPD, CNN)
- https://github.com/PaulLerner/deep_parkinson_handwriting (online handwriting DL)
- Kaggle notebooks on kmader/parkinsons-drawings (VGG/ResNet transfer learning; e.g. the dataset's own notebook section)

## B. Datasets

| Dataset | Subjects | Images | Data types | Classes | Availability | Reference |
|---------|----------|--------|------------|---------|--------------|-----------|
| Parkinson's Drawings ("PD Drawings", Zham et al.) | 62 (31 PD / 31 healthy) | 204 unique (spiral + wave, 72 train + 30 test per type as released on Kaggle) | spiral, wave | healthy, parkinson | Kaggle: https://www.kaggle.com/datasets/kmader/parkinsons-drawings | Zham et al., IEEE EMBC 2016; Zham et al., *Front. Neurol.* 2017, 8:435 (https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2017.00435/full) |
| HandPD | 92 (74 PD / 18 healthy) drawn from a 156-subject clinical cohort | 736 micrographs (~700×700 JPG, 4 per subject) | spiral, meander | control, patients | UNESP: https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/ | Pereira et al., 2016, *Computer Methods and Programs in Biomedicine* 136:79–88 |
| NewHandPD | 31 PD / 31 healthy | 1056 (static color images + dynamic signals) | spiral, meander (+signature, character) | healthy, patients | UNESP (same lab page) | Pereira et al. (NewHandPD paper; citation string in arXiv:2111.14781 reference list) |
| PaHaW | 37 PD / 38 healthy | online handwriting (8 tasks/rep.) | sentence, word, spiral, graph | PD, healthy | request from authors (Masaryk Univ.) | Drotár et al., 2016; kinematic analysis in Drotár et al., 2014, CMPB 117(3):405–411 |
| UCI Parkinson Disease Spiral Drawings (digitized tablet) | 62 people w/ PD + 15 healthy | kinematic time series + derived images | spiral | PD, healthy | http://archive.ics.uci.edu/dataset/395/ | UCI ML Repository entry |

## C. Notes / expectations

- Reported test accuracies on the Kaggle Parkinson's Drawings release cluster
  around **77–97%** depending on model, augmentation and drawing type; the
  **wave** type is consistently easier than the **spiral** type.
- Most common backbones in this niche: **VGG16/VGG19, ResNet18/50/101,
  MobileNet, EfficientNet, custom small CNNs**, recently **ViT**.
- No peer-reviewed work we could find performs a *systematic corruption
  robustness* evaluation (JPEG/blur/noise/contrast/rotation at multiple
  severities) on these datasets — this is the gap our framework fills.
- HandPD is heavily class-imbalanced (74 PD vs 18 healthy subjects, 4:1 in
  images), so balanced metrics are mandatory when using it.
