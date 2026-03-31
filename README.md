# Freehand 3D Ultrasound Spine Reconstruction — DCLNet Reproduction

> Reproduction and extension of **DCL-Net** (Guo et al., MICCAI 2020) for sensorless freehand 3D ultrasound reconstruction of the **lumbar spine**, with a dual-head output architecture, spatial attention, and a Leave-One-Patient-Out evaluation protocol.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Sensor-free freehand 3D ultrasound reconstruction infers the 6-DOF relative pose between consecutive B-mode frames — no external tracker required. This repository reproduces DCL-Net on a lumbar spine dataset and introduces:

- **Dual-head output layer** — separate linear projections for translation (`fc_trans`) and rotation (`fc_rot`), with a 3× higher learning rate for the rotation head
- **Spatial attention module** — re-weights intermediate feature maps toward anatomically informative regions (lamina edges, bony structures)
- **LOPO cross-validation** — rigorous Leave-One-Patient-Out protocol on 7 sequences across multiple scan protocols (transverse, longitudinal, oblique)

---

## Reconstructions

### Case 0008

| Ground Truth | Predicted |
|:---:|:---:|
| ![GT case0008](assets/case0008_gt.gif) | ![Pred case0005](assets/case0005_pred.gif) |

> Green = ground truth trajectory · Red = predicted trajectory

---

## Interactive Demo

Explore 3D reconstructions, accumulated DOF plots, and per-fold metrics interactively:

[![Open In Colab](https://colab.research.google.com/drive/14hf23nUYgeKx4GA_NiJECnUwLP13dHZC)

The notebook includes:
- 3D frame-corner trajectory visualisation (Plotly, rotatable)
- Accumulated DOF plots (GT vs prediction) across all 6 DOFs
- Per-fold LOPO results summary

---

## Results

LOPO evaluation on held-out patient `08_JA_June_16` (6 scans per fold):

| Fold | Scans | Dist (mm) | Drift (mm) | T-MSE (mm²) | R-MSE (deg²) |
|:----:|:-----:|:---------:|:----------:|:-----------:|:------------:|
| F1   | 6     | 52.5 ± 18.4 | 93.6 ± 33.3  | 11.14 | 0.0116 |
| F2   | 6     | 48.7 ± 15.5 | 85.6 ± 24.2  | 11.16 | 0.0116 |
| F3   | 6     | 58.2 ± 26.6 | 104.3 ± 50.6 | 11.12 | 0.0124 |
| F4   | 6     | 67.8 ± 34.4 | 121.4 ± 65.9 | 11.12 | 0.0135 |
| F5   | 6     | 57.8 ± 20.9 | 104.7 ± 37.5 | 11.14 | 0.0115 |
| **All** | **30** | **57.0 ± 24.9** | **101.9 ± 46.4** | **11.14** | **0.0121** |

**Key finding:** T-MSE is near-constant across all folds (11.12–11.16 mm²), indicating the model converges to a generic motion prior under MSE-only supervision — consistent with the original paper's motivation for the case-wise correlation loss. The correlation loss is implemented (see `losses.py`) but not yet connected to the training loop in this ablation.

---

## Architecture

```
Diff frames (N−1 ch)
        │
   ┌────▼─────────────────────────┐
   │  3D ResNeXt-50 backbone      │
   │  (conv1 → layer1 → layer2)   │
   └────────────────┬─────────────┘
                    │
           ┌────────▼────────┐
           │ Spatial attention│
           │  BN→Conv→ReLU   │
           │  →Conv→BN→Sigmoid│
           └────────┬─────────┘
                    │  Global avg pool → (B, 128)
           ┌────────┴──────────┐
           │                   │
      ┌────▼────┐         ┌────▼────┐
      │ fc_trans│         │  fc_rot │  ← 3× LR
      │128→3(N-1)         │128→3(N-1)
      └────┬────┘         └────┬────┘
           └────────┬──────────┘
                    │  interleave
            [tx,ty,tz,rx,ry,rz] × (N−1)
```

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
├── assets/                    # GIFs and figures for README
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
| `warmup_epochs` | 9999 | Curriculum disabled (long windows from epoch 1) |

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
- [ ] Geometric corner-point distance + drift penalty in training loop
- [ ] Larger dataset (multi-patient LOPO)


---

## Acknowledgements

This work is part of an MSc thesis at École de technologie supérieure, Montréal, supported by the LATIS Lab.

---

<p align="center">
  <sub>Ihssene Brahimi · ÉTS · 2025-2027</sub>
</p>
