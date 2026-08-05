**Cortico-Cerebellar Predictive Module for Robust Cardiac MRI Segmentation**

Official implementation of:
*Cerebellar-Inspired Predictive Module Improves Robustness of Recurrent Segmentation Network on Noisy and Undersampled Cardiac MRI* Ekaterina Kostina, Anastasia Sinitsyna, Mikhail Slotvitsky, Valeriya A. Tsvelaya

**Overview**

Left atrium (LA) segmentation from cardiac MRI is a critical step in catheter ablation planning for atrial fibrillation. Standard recurrent architectures are vulnerable to the noise, artifacts, and incomplete spatial coverage typical of clinical MRI. This repository implements a hybrid segmentation architecture inspired by cortico-cerebellar interactions:

1)Convolutional encoder — extracts per-slice bottleneck features
2)Cortical recurrent module (RNN) — models inter-slice temporal dependencies
3)Cerebellar predictive module — predicts future encoder features across multiple temporal horizons (τ ∈ {1, 2, 3, 4}) and generates a corrective feedback signal, improving robustness to image degradation without sacrificing mean accuracy.

The model is compared against a non-predictive readout baseline and against nnU-Net v2 as a state-of-the-art baseline, on both clean and degraded (Gaussian noise + slice subsampling) data, plus an independent external validation cohort.

**Requirements**

Python 3.8+
PyTorch 2.5.1 (CUDA 12.1, cuDNN 9.1.0)

**Data**

ATRIA Segmentation Challenge 2018 — publicly available, used for training and primary validation (80/20 patient-level split). Download: https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/
