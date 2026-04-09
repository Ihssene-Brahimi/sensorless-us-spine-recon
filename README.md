# Freehand 3D Ultrasound Spine Reconstruction — DCLNet Reproduction

> Reproduction and extension of **DCL-Net** (Guo et al., MICCAI 2020) for sensorless freehand 3D ultrasound reconstruction of the **spine**, with a dual-head output architecture, spatial attention, and a Leave-One-Patient-Out evaluation protocol.

[![Open Demo](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/14hf23nUYgeKx4GA_NiJECnUwLP13dHZC#scrollTo=e8c7ac2d])
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Sensorless freehand 3D ultrasound reconstruction infers the 6-DOF relative pose between consecutive B-mode frames — no external tracker required. This repository reproduces DCL-Net on a lumbar/thoracic spine dataset and introduces:

- **Dual-head output layer** — separate linear projections for translation (`fc_trans`) and rotation (`fc_rot`), with a 3× higher learning rate for the rotation head
- **LOPO cross-validation** — Leave-One-Patient-Out protocol on 7 sequences 

---

## Reconstructions

[![Interactive 3D](https://img.shields.io/badge/Interactive_3D-View_Reconstruction-185FA5?style=for-the-badge)](https://ihssene-brahimi.github.io/sensorless-us-spine-recon/spine_case_8_gt.html)

### Case 0008

| Ground Truth | Predicted |
|:---:|:---:|
| ![GT case0008](assets/case0008_gt.gif) | ![Pred case0005](assets/case0008_pred.gif) |

> Green = ground truth trajectory · Red = predicted trajectory

---

## Repository Structure

```
.
├── networks/
│   ├── mynet.py               # Dual-head ResNeXt architecture
│   ├── resnext.py             # Original ResNeXt (single head, for comparison)
│   └── models.py              # Model factory
├── losses.py                  # MSE, correlation, geometric, drift losses
├── transforms.py              # Data augmentation transforms
├── reconstruction_v1.py       # Core training/validation loop
├── train-reconstruction-v1-lopo.py   # LOPO cross-validation entry point
├── test_reconstruction.py     # Evaluation and visualisation
├── mytools.py                 # Geometry utils (numpy + torch), sampling
├── data-spine-best-seven.json # Dataset config (7 sequences)
├── assets/                   
│   ├── case0008_gt.gif
│   └── case0008_pred.gif
└── README.md
```

---

## Training

```bash
# LOPO cross-validation (Leave-One-Patient-Out)
python train-reconstruction-v1-lopo.py
```

Key config options (edit top of `train-reconstruction-v1-lopo.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `neighbour_num` | 10 | Number of input frames |
| `lr` | 1e-5 | Base learning rate |
| `lr_rot_scale` | 3.0 | Rotation head LR multiplier |
| `w_rot` | 2.0 | Rotation loss weight |
| `smooth_sigma` | 1.0 | GT smoothing (Gaussian) |

MLflow tracking is enabled by default. Launch the UI with:

```bash
mlflow ui --port 5000
```

---

## Evaluation

```bash
python test_reconstruction.py \
    --model_path Results/<run_id>/best_model.pth \
    --data_json data-spine-best-seven.json \
    --output_dir Results/<run_id>/test_output
```

Outputs per scan: 3D reconstruction plot, accumulated DOF curves (GT vs pred), and a summary table with dist / drift / T-MSE / R-MSE.

---

## Planned Extensions

- [ ] Add lamina landmark loss (annotations collected)
- [ ] Larger dataset



---

<p align="center">
  <sub>Ihssene Brahimi · ÉTS · 2025-2027</sub>
</p>
