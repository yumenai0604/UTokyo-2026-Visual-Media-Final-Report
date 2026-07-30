# Third-Party Components

This repository is supplementary material and depends on third-party software
and data that are not redistributed here.

## SARDet-100K and MSFA

- Project: <https://github.com/zcablii/SARDet_100K>
- Paper: Li et al., *SARDet-100K: Towards Open-Source Benchmark and ToolKit for
  Large-Scale SAR Object Detection*, NeurIPS 2024 Spotlight.

The detector architecture, upstream training/evaluation framework, pretrained
weights, and dataset remain subject to the terms published by their respective
authors and distributors. The two files under `configs/` are report-specific
evaluation configurations derived from the upstream MSFA setup; they are
included to disclose the exact tested inference settings.

## OpenMMLab and Python packages

MMDetection, MMCV, MMEngine, PyTorch, Kymatio, NumPy, SciPy, OpenCV, and
Matplotlib retain their own licenses. See the upstream projects for their
license texts and attribution requirements.

The MIT License in this repository applies only to the original supplementary
scripts and documentation authored for this report. It does not relicense any
third-party code, model, weight, or dataset.
