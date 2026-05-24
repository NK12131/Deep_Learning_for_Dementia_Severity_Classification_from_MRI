<div align="center">

# Dementia Detection & Severity Prediction
### Deep Learning Classification of MRI Brain Scans into Four Clinical Stages

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Azure ML](https://img.shields.io/badge/Azure%20ML-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/en-us/products/machine-learning)
[![Accuracy](https://img.shields.io/badge/Accuracy-87%25-22863a?style=flat-square)]()
[![F1](https://img.shields.io/badge/Macro%20F1-0.87-22863a?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

*Custom CNN + DCGAN augmentation + Grad-CAM explanations — 87% accuracy on four-class severity classification.*

[Overview](#overview) · [Dataset](#dataset) · [Architecture](#architecture) · [Results](#results) · [Grad-CAM](#grad-cam) · [Usage](#usage)

</div>

---

## Overview

Dementia affects **55 million people worldwide**, with cases projected to nearly triple by 2050. Automated MRI classification could support earlier diagnosis — but most models are black boxes that radiologists can't trust or act on.

This project addresses both accuracy and interpretability:

- **87% accuracy** classifying brain MRIs into four severity levels
- **Grad-CAM heatmaps** identifying which brain regions drive each prediction
- **DCGAN augmentation** resolving severe class imbalance in the training data
- GPU-accelerated training on **Azure ML**

### Severity Classes

| Class | Label | Clinical Description |
|:---:|:---|:---|
| 0 | Non-Demented | No cognitive impairment |
| 1 | Very Mild Demented | Slight memory lapses; functionally independent |
| 2 | Mild Demented | Noticeable memory loss; difficulty with complex tasks |
| 3 | Moderately Demented | Severe impairment; significant loss of independence |

---

## Dataset

**Source:** [Kaggle — MRI Dementia Dataset (No Data Leak)](https://www.kaggle.com/datasets/matthewhema/mri-dementia-augmentation-no-data-leak/data)

| | |
|:---|:---|
| Original size | 6,400 structural MRI images |
| Post-augmentation | 6,400 images, **1,600 per class** |
| Augmentation method | Per-class DCGAN |

### Why DCGAN Instead of Standard Augmentation?

The original dataset was heavily skewed — Moderate Demented had a fraction of the Non-Demented samples. Flipping or rotating the minority class doesn't add structural diversity; it just copies the same scans.

A DCGAN was trained separately for each underrepresented class, generating structurally novel synthetic MRIs that preserve class-specific neuroanatomical patterns. This brought every class to 1,600 balanced samples without leaking inter-class spatial features.

---

## Architecture

### Custom CNN *(Best Model)*

```
Input (grayscale MRI)
  │
  ├─ Conv Block 1: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
  ├─ Conv Block 2: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
  └─ Conv Block 3: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
  │
  ├─ Dense(256) → Dropout
  └─ Dense(128) → Dropout
  │
  Softmax → 4 classes
```

**Design rationale:**
- Batch normalization stabilizes training across varying MRI intensity distributions
- Dropout throughout prevents overfitting on limited clinical data
- Built from scratch for grayscale MRI input — this is what made it outperform every ImageNet-pretrained model tested

### DCGAN for Synthetic Augmentation

One DCGAN per class. Generator outputs synthetic MRIs from a noise vector; discriminator learns to distinguish real from synthetic. Training per-class ensures synthetic scans carry class-specific morphological patterns rather than generic brain structure.

```
Generator                         Discriminator
─────────────────────             ─────────────────────────
Noise z ~ N(0,1)                  Real or synthetic MRI
  → Dense → Reshape                 → Conv2D → LeakyReLU
  → TransposeConv + BN + ReLU       → Conv2D + BN + LeakyReLU
  → TransposeConv + BN + ReLU       → Conv2D + BN + LeakyReLU
  → TransposeConv + Tanh            → Dense → Sigmoid
```

### Architectures Benchmarked

| Model | Accuracy | Notes |
|:---|:---:|:---|
| **Custom CNN** | **87%** | Designed for grayscale MRI — best performer |
| InceptionV3 | 57% | Multi-scale features; ImageNet domain gap |
| DenseNet | 55% | Dense connections helped marginally |
| ResNet | 49% | Residual skip connections; poor MRI texture fit |
| EfficientNet | 28% | Severe underfitting on grayscale input |

The pattern is consistent: ImageNet pretraining provides no benefit here. Natural image features don't generalize to structural MRI textures. A purpose-built architecture wins by a wide margin.

---

## Results

| Model | Precision | Recall | F1 | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **Custom CNN** | **0.87** | **0.87** | **0.87** | **0.87** |
| InceptionV3 | 0.57 | 0.57 | 0.56 | 0.57 |
| DenseNet | 0.55 | 0.55 | 0.55 | 0.55 |
| ResNet | 0.49 | 0.49 | 0.49 | 0.49 |
| EfficientNet | 0.15 | 0.28 | 0.16 | 0.28 |

**Notable:** The Very Mild Demented class — the most clinically important stage for early intervention — achieved the highest per-class accuracy. The model is most reliable exactly where it matters most.

---

## Grad-CAM

High accuracy alone isn't enough for clinical use. Radiologists need to understand *why* a model flagged a scan before acting on it.

Grad-CAM backpropagates class gradients to the final convolutional layer, producing a spatial heatmap of which regions most influenced the prediction. These are overlaid on the original MRI scan.

### Activation Patterns by Severity

| Severity | Activation Pattern | Clinical Interpretation |
|:---|:---|:---|
| Non-Demented | Diffuse, minimal | Intact structure; no focal pathology |
| Very Mild | Subtle medial temporal focus | Early hippocampal and entorhinal changes |
| Mild | Concentrated hippocampal | Visible atrophy in memory-consolidation structures |
| Moderate | Widespread cortical | Global neurodegeneration across frontal and temporal lobes |

Activations align with known neuroanatomical markers of Alzheimer's progression — confirming the model is responding to genuine pathological signals rather than imaging artifacts or scanner variation.

The result: instead of a black-box severity label, a radiologist receives a spatially explicit explanation they can verify against expected atrophic changes before acting on the prediction.

---

## Challenges

| Challenge | Solution |
|:---|:---|
| Severe class imbalance (Moderate critically underrepresented) | Per-class DCGAN augmentation to 1,600 images each |
| ImageNet → MRI domain gap | Custom CNN designed for grayscale MRI input |
| DCGAN GPU requirements exceeding local hardware | Azure ML cloud compute |
| Black-box predictions limiting clinical trust | Grad-CAM spatial heatmaps |
| Overfitting risk on limited medical imaging data | Batch normalization + dropout throughout |

---

## Future Work

- **StyleGAN** — higher-fidelity synthetic MRI generation
- **3D volumetric CNNs** — process full MRI volumes rather than 2D axial slices
- **Multimodal fusion** — combine MRI with cognitive scores and biomarkers
- **Federated learning** — multi-hospital training without sharing patient data
- **Clinical deployment** — web-based diagnostic decision-support tool

---

## References

- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Radford et al. (2015). *Unsupervised Representation Learning with Deep Convolutional GANs.* arXiv:1511.06434.
- [Deep Learning in Dementia Diagnosis — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1110016822005191)
- [Neural Networks in Neuroimaging — Nature](https://www.nature.com/articles/s41467-022-31037-5)
- [Deep Learning for MRI Analysis — IEEE](https://ieeexplore.ieee.org/document/9587953)

---

<div align="center">

**Dataset:** [Kaggle MRI Dementia](https://www.kaggle.com/datasets/matthewhema/mri-dementia-augmentation-no-data-leak) · **Compute:** [Azure ML](https://azure.microsoft.com/en-us/products/machine-learning) · **Author:** [Nithin Kumar](https://nithinkumar.vercel.app)

*Applied Data Science research project — explainable AI in clinical neuroimaging.*

</div>