# UTokyo-2026-Visual-Media-Final-Report
Supplementary scripts for the Visual Media course report:
**"Failure Analysis and Post-Processing Improvement of MSFA on SARDet-100K"**

University of Tokyo, Graduate School of Engineering, 2025–2026

---

## Original Repository

All model code, pre-trained weights, and dataset tools are from the official SARDet-100K release:

**[SARDet-100K — Official GitHub](https://github.com/zcablii/SARDet_100K)**

> Li, Y., et al. (2024). *SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection.* NeurIPS 2024 Spotlight.

Please follow the original repository's instructions to set up the environment, download the dataset, and obtain the pre-trained MSFA weights before using the scripts below.

---

## Scripts in This Repository

| File | Purpose |
|---|---|
| `prepare_data.py` | Generate a speckle-noise-degraded test set from the clean SARDet-100K test images |
| `run_eval.py` | MMDetection inference config — Hard-NMS (baseline) |
| `run_eval_softnms.py` | MMDetection inference config — Soft-NMS |

---

## Usage

### 1. Generate the Noisy Test Set

`prepare_data.py` applies physically-correct multiplicative speckle noise (Gamma distribution, ENL=L) to all images in the SARDet-100K test set. The noise is sampled as a single 2D realization and broadcast across all channels, consistent with the grayscale nature of SAR imagery.

```bash
python prepare_data.py
```

**Key parameters** (edit at the top of the script):

```python
VAL_DIR  = '/path/to/SARDet_100K/JPEGImages/test'   # clean test images
OUT_DIR  = '/path/to/SARDet_100K/JPEGImages/test_speckle_L4'  # output directory
L        = 4    # equivalent number of looks (ENL); higher = less noise
```

The noise model follows:

```
I_noisy = I_clean * n,   n ~ Gamma(shape=L, scale=1/L)
```

where `n` is a single (H, W) noise map broadcast to all channels.

---

### 2. Run Inference

Use the provided config files with MMDetection's `test.py`. Replace paths as needed.

**Hard-NMS (baseline):**
```bash
python tools/test.py run_eval.py /path/to/checkpoint.pth
```

**Soft-NMS:**
```bash
python tools/test.py run_eval_softnms.py /path/to/checkpoint.pth
```

The configs assume the MSFA wavelet backbone (Faster-RCNN + ResNet50, 82-channel input). Checkpoint path and dataset paths should be updated inside each config file.

---

## Experimental Results

| Condition | mAP | AP@50 | AP@75 |
|---|---|---|---|
| Hard-NMS (Clean) | 0.511 | 0.839 | 0.547 |
| Soft-NMS (Clean) | 0.529 | 0.848 | 0.578 |
| Hard-NMS (L=4)   | 0.162 | 0.282 | 0.169 |
| Soft-NMS (L=4)   | 0.150 | 0.265 | 0.154 |

Noise simulation: Gamma multiplicative speckle, ENL=4, single-channel broadcast.
