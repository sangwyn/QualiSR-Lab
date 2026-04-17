# Reduced-Reference IQA for Super-Resolution

## Overview

This project studies which features extracted from Low-Resolution (LR) and Super-Resolution (SR) images are most informative for Image Quality Assessment (IQA).

The proposed pipeline is:

1. **Prepare labels and features**  
   Compute image features and attach normalized quality labels.

2. **Train regressors**  
   Fit regression models on the resulting tabular data to obtain a simple Reduced-Reference (RR) quality metric.

3. **Analyze feature importance and correlation**  
   Evaluate feature importance and compute PLCC/SRCC to identify the most informative features for SR quality assessment.

The sections below describe the required data format and the workflow.

---

## Dataset Structure

Example of a valid dataset layout:

```
dataset/
├── gt/
│   ├── 0000001.png        # GT images, shape: (H, W, 3)
│   └── ...
├── lr/
│   ├── 0000001.png        # LR images, shape: (H/scale, W/scale, 3)
│   └── ...
├── heatmaps/
│   ├── sr_method_1/
│   │   ├── 0000001.npy.gz # Artifact masks, shape: (H, W, 1)
│   │   └── ...
│   ├── ...
│   └── sr_method_N/
│       ├── 0000001.npy.gz
│       └── ...
├── sr/
│   ├── sr_method_1/
│   │   ├── 0000001.png    # SR images, shape: (H, W, 3)
│   │   └── ...
│   ├── ...
│   └── sr_method_N/
│       ├── 0000001.png
│       └── ...
└── ref/
    ├── ref_method_1/
    │   ├── 0000001.png    # Reference (quasi-GT) images, shape: (H, W, 3)
    │   └── ...
    ├── ...
    └── ref_method_M/
        ├── 0000001.png
        └── ...
```

SR images must have normalized quality scores in the range `[0, 1]`:

```csv
labels.csv

test_case,method,score_norm
0000001,sr_method_1,0.72
0000001,sr_method_2,0.25
0000002,sr_method_1,0.59
...
```

We provide a free sample dataset that follows these guidelines. Please look at `dataset/readme.md` for details on downloading and using it. Sample features in this repository are precomputed on this data.

---

## Workflow

### Step 0 (optional): Prepare reference images

Produce [RLFN](https://github.com/bytedance/RLFN) / [SPAN](https://github.com/zononhzy/SPAN) / bicubic images for LR + SR pairs (used to compute FR metrics).

```bash
python scripts/make_reference.py \
  --lr-dir data/lr \
  --sr-dirs PASD=data/PASD SUPIR=data/SUPIR RealESRGAN=data/RealESRGAN \
  --out-root data/ \
  --refs bicubic rlfn span \
  --scale 4 \
  --rlfn-script realtime_sr/RLFN/inference-RLFN.py \
  --rlfn-ckpt realtime_sr/RLFN/rlfn-tuned-4x.pth \
  --span-script realtime_sr/SPAN/inference-SPAN.py \
  --span-ckpt realtime_sr/SPAN/span-tuned-4x.pth
```

### Step 1: Compute image features

Compute FR / NR / [VGG](https://arxiv.org/abs/1409.1556) / [ResNet](https://arxiv.org/abs/1512.03385) / [SigLIP](https://arxiv.org/abs/2303.15343) features for SR images and save them into a single CSV file.

SR methods are passed as `METHOD=DIR`.  
Reference image filenames are expected in the format:

```text
<sr_stem>@<sr_method>@<ref_name>.<ext>
```

```bash
python scripts/get_image_features.py \
  --sr-dirs PASD=data/PASD SUPIR=data/SUPIR RealESRGAN=data/RealESRGAN \
  --gt-dir data/gt \
  --lr-dir data/lr \
  --ref-dirs bicubic=data/bicubic rlfn=data/RLFN span=data/SPAN \
  --features fr,nr,vgg,resnet,siglip \
  --output features/image_features.csv \
  --device cuda
```

---

### Step 2: Apply PCA to high-dimensional features

Apply Principal Component Analysis (PCA) to high-dimensional feature blocks such as `vgg_*` and `resnet_*` in CSV files produced in Step 1.

```bash
python scripts/apply_pca.py \
  --input features/image_features.csv \
  --n-components 5 10 25 50 75 \
  --test-size 0.2 \
  --split-seed 42 \
  --output-dir features/pca
```

---

### Step 3: Compute artifact-mask statistics

Compute summary statistics for heatmaps stored as `.npy`, `.npy.gz`, or compatible compressed files.  
Input directories can be passed as `PREFIX=DIR` to ensure stable sample naming.

```bash
python scripts/compute_statistics.py \
  --heatmap-dirs PASD=data/heatmaps/PASD SUPIR=data/heatmaps/SUPIR RealESRGAN=data/heatmaps/RealESRGAN \
  --output stats/grounding_stats@coarse.csv \
  --percentiles 5 95 \
  --area-thresholds 0 0.5 0.75
```

---

### Step 4: Fit regressors and analyze results

The main notebook for experiments is `regressors.ipynb`. It trains regressors, evaluates them, and visualizes:

- feature importances,
- PLCC/SRCC correlations,
- comparisons across feature groups and model settings.

The first notebook cell describes the workflow for running experiments individually or in batches.

Example outputs:

<p float="left">
  <img src="plots/example@pca5/all_models_importances.png" alt="Feature importances" width="700"/>
  <img src="plots/example@pca5/correlations.png" alt="Correlations" width="550"/>
</p>

---

## Feature Types

This section summarizes the feature groups used in the pipeline.

### No-Reference (NR) metrics

NR metrics are widely used in SR-IQA because they do not require a perfect high-resolution reference image. Their main limitation is that they ignore information available in the input LR image, which may cause them to miss or even reward artifacts introduced by SR models.

Recommended NR metrics in this project, based on results from the [SR Metrics Benchmark](https://videoprocessing.ai/benchmarks/super-resolution-metrics.html):

- [Q-Align](https://github.com/Q-Future/Q-Align)
- [MUSIQ](https://github.com/anse3832/MUSIQ)
- [ARNIQA](https://github.com/miccunifi/ARNIQA)
- [UNIQUE](https://github.com/zwx8981/UNIQUE)
- [PaQ2PiQ](https://github.com/baidut/paq2piq)

These metrics are computed through the [**PyIQA**](https://github.com/chaofengc/IQA-PyTorch) interface, so the list can be changed easily.

---

### Full-Reference (FR) metrics

FR metrics are not always ideal for SR-IQA because they assume access to a perfect reference image. Still, they provide useful information about fidelity.

When true GT images are unavailable, the project uses **quasi-GT** references: images obtained by upscaling the LR input with methods that are faithful to the LR image and do not introduce strong hallucinated content.

Reference upscaling methods used here:

- bicubic interpolation
- [SPAN](https://github.com/zononhzy/SPAN)
- [RLFN](https://github.com/bytedance/RLFN)

Recommended FR metrics in this project, based on results from the [SR Metrics Benchmark](https://videoprocessing.ai/benchmarks/super-resolution-metrics.html):

- [LPIPS-VGG](https://github.com/richzhang/perceptualsimilarity)
- [STLPIPS-VGG](https://github.com/abhijay9/ShiftTolerant-LPIPS)
- [PieAPP](https://github.com/prashnani/PerceptualImageError)
- [AHIQ](https://github.com/IIGROUP/AHIQ)
- [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [SSIM](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html)

These metrics are also computed through [**PyIQA**](https://github.com/chaofengc/IQA-PyTorch).

---

### Pretrained encoder features (+ PCA)

Feature embeddings from pretrained encoders can capture semantic and perceptual information not covered by classical IQA metrics.

This project uses features extracted from:

- [VGG](https://arxiv.org/abs/1409.1556)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [SigLIP](https://arxiv.org/abs/2303.15343)

Because these embeddings are often high-dimensional, Principal Component Analysis (PCA) can be applied before training regressors.

---

### Artifact-mask statistics

Artifacts are common in modern deep-learning-based SR models. The working hypothesis of this project is:

> Artifact-related information provides useful signals for assessing generated image quality.

An artifact mask is a single-channel tensor with values in the range `[0, 1]`.  
Masks for SR images must be computed beforehand.

The project extracts the following summary statistics from artifact masks:

- min
- max
- mean
- median
- std
- percentiles
- thresholded artifact area


<!-- ---

## Summary

This repository provides a practical pipeline for building and analyzing reduced-reference IQA metrics for super-resolution. It combines:

- NR metrics,
- FR metrics,
- pretrained encoder embeddings,
- artifact-mask statistics,

and uses regression analysis plus correlation metrics to determine which features are most informative for SR quality prediction. -->
