# Dataset

Source image pool: [SR-Ground](https://huggingface.co/datasets/Divotion/SR-Ground).

## 🔎 Overview

This is a sample dataset based on open IQA datasets: [FLIVE](https://github.com/niu-haoran/FLIVE_Database), [KonIQ10k](https://database.mmsp-kn.de/koniq-10k-database.html), [AVA](https://github.com/imfing/ava_downloader). We picked 40 ground-truth images based on K-Means clusterization (10 images for each of the 4 centroids). For each of these images, we provide both SR ([PASD](https://github.com/yangxy/PASD)/[SUPIR](https://github.com/Fanghua-Yu/SUPIR)/[RealESRGAN](https://github.com/xinntao/real-esrgan)) and reference ([RLFN](https://github.com/bytedance/RLFN)/[SPAN](https://github.com/zononhzy/SPAN)/bicubic) images and also artifact masks in `heatmaps/` subdirectory. Each SR image has a Mean Opinion Score assigned to it in `labels.csv`.

---

## ⚙️ How to use

Dataset is available to download for free on [Hugging Face](https://huggingface.co/datasets/onryabinin/QualiSR-Set120) or [GDrive](https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing). Below are the commands to download and unpack all the images.

```bash
hf download onryabinin/QualiSR-Set120 --repo-type dataset --local-dir dataset
```

Alternatively:

```bash
gdown 'https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing'
unzip grounding_dataset.zip -d dataset/
```

---

## 💿 Dataset Structure

Below is an example of a valid dataset layout. When using your own data, it should resemble this structure like the dataset we present above. Our dataset includes artifact mask heatmaps but they are not required for the pipeline.

```
dataset/
├── hr/
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
    │   ├── 0000001.png    # Reference (pseudo-GT) images, shape: (H, W, 3)
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

---

## 🔑 License

This dataset combines data derived from multiple third-party sources.

* Content originating from the included third-party datasets follows the licensing terms of those original sources. In the current project setup, these components are treated as MIT-compatible for redistribution and research use where applicable.

If you plan to redistribute or use this dataset in downstream work, you should verify that your use complies with the licenses of all original data sources.
