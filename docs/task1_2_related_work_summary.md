# Task 1.2 — Robustness / Benchmark Evaluation Frameworks: Related-Work Summary

*Compiled 2026-09-08 from web searches; all references verified against real
pages (arXiv, publisher, or GitHub).*

## 1. Robustness evaluation in general computer vision

The canonical methodology is **ImageNet-C** (Hendrycks & Dietterich, ICLR 2019,
[arXiv:1903.12261](https://arxiv.org/abs/1903.12261),
[code](https://github.com/hendrycks/robustness),
[data](https://zenodo.org/records/2235448)): 15 corruption families (noise,
blur, weather, digital — incl. Gaussian noise, Gaussian blur, JPEG compression,
contrast/brightness changes) applied at **5 increasing severity levels**, with
the **mean Corruption Error (mCE)** as headline metric: a model's average error
across corruptions/severities normalized by a fixed reference model (AlexNet).
The same paper introduced ImageNet-P (perturbation sequences, mean Flip Rate).
**RobustBench** (Croce et al., LDMI 2021) standardized adversarial + corruption
robustness leaderboards ([robustbench.github.io](https://robustbench.github.io)).

**Takeaway for our framework:** the corruption-family × 5-severity design and a
normalized aggregate score are the community standard; we adapt them to
handwriting/drawing images, where "corruption" corresponds to realistic image
degradations of photographed/scanned drawings (compression, blur, sensor noise,
exposure changes, small rotation).

## 2. Robustness in medical imaging

- **MedMNIST-C**: 12 corrupted benchmark datasets for robust medical image
  classification with **five severity levels** per corruption and augmentation
  APIs ([Univ. of Bamberg page](https://www.uni-bamberg.de/en/ai/chair-of-explainable-machine-learning/software-datasets/dataset-medmnist-c-12-corrupted-benchmark-datasets-and-augmentation-apis-for-robust-medical-image-classification/)).
- **ROOD-MRI** (Boone et al., *NeuroImage* 2023,
  [paper](https://www.sciencedirect.com/science/article/pii/S1053811923004408)):
  benchmarking robustness of MRI segmentation networks to corruptions/artifacts;
  finds CNNs highly susceptible to distribution shifts, simple augmentations help.
- **Javed et al.**, "Robustness in Deep Learning Models for Medical
  Diagnostics: A Survey", *Artificial Intelligence Review* 57, 2024
  ([link](https://link.springer.com/article/10.1007/s10462-024-11005-9)).
- **Shen et al.**, 2025 (PubMed
  [40587343](https://pubmed.ncbi.nlm.nih.gov/40587343/)): improving robustness
  and reliability of medical image classification against unexpected image
  corruptions and noise perturbations encountered after deployment.
- A recurring, consistent finding: **medical DL models degrade substantially
  under mild corruptions**, and evaluation on clean test data alone
  overestimates real-world performance.

## 3. Robustness / image quality in PD handwriting-drawing detection

PD detection from drawings is dominated by architecture papers (accuracy on
clean data; see `task1_1_models_datasets.md`). Searches for a systematic
corruption-robustness benchmark on spiral/wave/meander PD datasets returned **no
direct match** — existing work touches robustness only indirectly (image
pre-processing, denoising, augmentation as accuracy tricks, e.g. the Tilburg
thesis on pre-processing effects,
[arno.uvt.nl](http://arno.uvt.nl/show.cgi?fid=169234), reports ≤93% with
pre-processing variations). **This is the research gap our paper addresses.**

## 4. XAI and its stability

- **Grad-CAM** (Selvaraju et al., ICCV 2017, pp. 618–626,
  [arXiv:1610.02391](https://arxiv.org/abs/1610.02391), DOI
  10.1109/ICCV.2017.74; extended IJCV 2020, 128:336–359).
- **Sanity Checks for Saliency Maps** (Adebayo et al., NeurIPS 2018,
  [PDF](https://papers.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf)):
  saliency maps can look plausible even after model/label randomization → maps
  must be validated, not just visualized. Notably, Grad-CAM *passes* the model
  parameter-randomization test ([revisit: arXiv:2110.14297](https://arxiv.org/abs/2110.14297)).
- Saliency-map (dis)similarity is commonly quantified with **Pearson/Spearman
  correlation, SSIM, and IoU of top-k% active pixels**; we adopt these for
  measuring XAI stability under corruption.

## 5. Takeaways for our framework (design decisions)

1. **Corruption families (6) × 5 severity levels**, mirroring ImageNet-C:
   JPEG compression (Q 90→10), Gaussian blur (k 3→11), Gaussian noise
   (σ 0.01→0.12), contrast (0.5→1.5), brightness (0.5→1.5), rotation (±5°→±25°,
   white fill matching paper background).
2. **Metrics**: accuracy + balanced accuracy, precision, sensitivity,
   specificity, F1, ROC-AUC on clean data; **relative degradation** per metric
   per corruption, averaged over severities (ImageNet-C style, mCE-adapted
   without a fixed AlexNet reference since no public baseline exists).
3. **XAI stability**: Grad-CAM for all three CNN backbones; Spearman rank
   correlation between each corrupted map and the clean map (primary), plus
   Pearson, SSIM, IoU(top-20%) as secondary.
4. **Aggregate robustness score**: weighted combination of relative accuracy
   drop (w=0.5), F1 drop (w=0.3) and XAI instability (w=0.2) —
   the *Parkinson Robustness Score* (PRS, 0–100, higher = more robust).

## 6. References

1. Hendrycks, D., Dietterich, T. "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations." ICLR 2019. https://arxiv.org/abs/1903.12261
2. Croce, F., et al. "RobustBench: A standardized adversarial robustness benchmark." LDMI 2021. https://robustbench.github.io
3. MedMNIST-C: 12 corrupted benchmark datasets for robust medical image classification. Univ. Bamberg. https://www.uni-bamberg.de/en/ai/chair-of-explainable-machine-learning/software-datasets/dataset-medmnist-c-12-corrupted-benchmark-datasets-and-augmentation-apis-for-robust-medical-image-classification/
4. Boone, L., et al. "ROOD-MRI: Benchmarking the robustness of deep learning segmentation networks to distribution shift." NeuroImage 280, 2023. https://www.sciencedirect.com/science/article/pii/S1053811923004408
5. Javed, H., et al. "Robustness in deep learning models for medical diagnostics: a survey." Artificial Intelligence Review 57, 2024. https://link.springer.com/article/10.1007/s10462-024-11005-9
6. Shen, X., et al. "Improving Robustness and Reliability in Medical Image Classification." 2025. https://pubmed.ncbi.nlm.nih.gov/40587343/
7. Selvaraju, R.R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017, 618–626. https://arxiv.org/abs/1610.02391
8. Adebayo, J., et al. "Sanity Checks for Saliency Maps." NeurIPS 2018, 9505–9515. https://papers.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf
9. Huang, Y., et al. "Early Parkinson's Disease Diagnosis through Hand-Drawn Spiral and Wave Analysis Using Deep Learning Techniques." Information 15(4):220, 2024. https://www.mdpi.com/2078-2489/15/4/220
10. Pereira, C.R., et al. "A new computer vision-based approach to aid the diagnosis of Parkinson's disease." Computer Methods and Programs in Biomedicine 136:79–88, 2016. https://www.sciencedirect.com/science/article/pii/S0169260716301894
11. Zham, P., et al. "Efficacy of Guided Spiral Drawing in the Classification of Parkinson's Disease." IEEE EMBC 2016, 5007–5010.
12. Zham, P., et al. "Distinguishing Different Stages of Parkinson's Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral." Frontiers in Neurology 8:435, 2017. https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2017.00435/full
13. Drotár, P., et al. "Analysis of in-air movement in handwriting: A novel marker for Parkinson's disease." Computer Methods and Programs in Biomedicine 117(3):405–411, 2014.
14. "Machine Learning for Real-Time, Automatic, and Early Diagnosis of Parkinson's Disease by Extracting Signs of Micrographia from Handwriting Images." ML4H 2021. https://arxiv.org/abs/2111.14781
15. HandPD dataset official page (Spiral/Meander zips). https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/
16. Kaggle: Parkinson's Drawings (K. Scott Mader release). https://www.kaggle.com/datasets/kmader/parkinsons-drawings
17. Frontiers in Medicine 2024. "Utilizing deep learning models in an intelligent spiral drawing classification system." https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1453743/full
18. Razaq, A., et al. "Deep Learning Framework for Early PD Detection." Diagnostics 15(21):2795, 2025. https://www.mdpi.com/2075-4418/15/21/2795
19. AIMS Mathematics 2024. "Modeling and diagnosis Parkinson disease by using hand drawing." https://www.aimspress.com/article/doi/10.3934/math.2024334
