# Evaluation Results

`summary.csv` records the final COCO metrics printed by MMDetection for the
experiments discussed in the report. Values were transcribed from the completed
evaluation logs; raw logs are excluded because they contain machine-specific
absolute paths and environment details.

All principal comparisons use the same released MSFA wavelet checkpoint and
the SARDet-100K test annotations. Only the image prefix and inference
configuration change between rows.

The "small-object image subset" contains every test image with at least one
annotation whose COCO area is below 1,024 pixels. Its overall mAP is not the
same quantity as COCO `AP_small`: images in that subset may also contain medium
or large annotations, which remain part of evaluation.

The Lee-filtered row documents an investigated but unsuccessful preprocessing
alternative. Its mAP (`0.148`) is below the noisy Hard-NMS baseline (`0.150`),
although AP50 is higher. It was therefore not presented as the final selected
improvement.

The Soft-NMS rows also use a lower score threshold and a higher per-image
detection cap. They are full inference-configuration comparisons rather than a
strict one-factor ablation.
