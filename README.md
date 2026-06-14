# Uncertainty-Aware Multimodal Clinical Decision System for Retinal Disease Diagnosis Using OCT Scans

🚀 **Patent Published:** Application No. **202641051634**

An end-to-end trust-aware AI system for retinal disease diagnosis from OCT scans, integrating multimodal deep learning, uncertainty estimation, explainable AI, and LLM-generated clinical rationales for reliable clinical decision support.

## Overview

This project processes raw OCT B-scan images and generates:

* Disease prediction
* Confidence scores
* Uncertainty estimation
* Grad-CAM heatmaps
* LLM-generated clinical explanations

The system performs 4-class retinal disease classification:

* **CNV**
* **DME**
* **DRUSEN**
* **NORMAL**

---

## Key Features

✅ Advanced preprocessing (CLAHE, vessel enhancement, intensity normalization)

✅ EfficientNetV2-S based visual feature extraction

✅ U-Net segmentation with GPU-accelerated skeletonization

✅ Extraction of vascular and topological biomarkers

✅ Bidirectional cross-attention feature fusion

✅ Uncertainty estimation using MC Dropout and Temperature Scaling

✅ Explainability using Grad-CAM

✅ Clinical rationale generation using LLaMA

✅ Interactive web interface for instant diagnostics

---

## Architecture

### Visual Stream

* EfficientNetV2-S
* Texture feature extraction

### Geometric Stream

* U-Net segmentation
* Skeletonization
* Biomarker extraction

### Fusion Module

* Bidirectional Cross-Attention

### Explainability & Reliability

* Grad-CAM
* MC Dropout
* Temperature Scaling

### Clinical Reasoning

* LLaMA-based rationale generation

---

## Performance Metrics

| Metric                           | Value      |
| -------------------------------- | ---------- |
| Accuracy                         | **96.79%** |
| Macro ROC-AUC                    | **0.9984** |
| Macro F1-Score                   | **0.9622** |
| Top-2 Accuracy                   | **99.17%** |
| Expected Calibration Error (ECE) | **0.0125** |

---

## Dataset

**Kermany OCT Dataset**

* 84,495 retinal OCT images (Kaggle Kermany2018 dataset)
* Four classes: CNV, DME, DRUSEN, NORMAL


---
