# Dataset

## 🔎 Overview

This is a sample dataset based on open IQA datasets: [FLIVE](https://github.com/niu-haoran/FLIVE_Database), [KonIQ10k](https://database.mmsp-kn.de/koniq-10k-database.html), [AVA](https://github.com/imfing/ava_downloader), [Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/). We picked 40 ground-truth images based on K-Means clusterization (10 images for each of the 4 centroids). For each of these images, we provide both SR ([PASD](https://github.com/yangxy/PASD)/[SUPIR](https://github.com/Fanghua-Yu/SUPIR)/[RealESRGAN](https://github.com/xinntao/real-esrgan)) and reference ([RLFN](https://github.com/bytedance/RLFN)/[SPAN](https://github.com/zononhzy/SPAN)/bicubic) images and also artifact masks in `heatmaps/` subdirectory. Each SR image has a Mean Opinion Score assigned to it in `labels.csv`.

---

## ⚙️ How to use

Dataset is available to download for free on [GDrive](https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing) or [GML server](https://titan.gml-team.ru:5003/sharing/hm3FpzpDp). Below are the commands to download and unpack all the images.

```bash
gdown 'https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing'
unzip grounding_dataset.zip -d dataset/
```

---

## 💿 Dataset Structure

Below is an example of a valid dataset layout. When using your own data, it should resemble this structure like the dataset we present above. Our dataset includes artifact mask heatmaps but they are not required for the pipeline.

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

---

## 🔑 License

This dataset combines data from multiple third-party sources:

- Data and images from the other included third-party datasets (except Waterloo Exploration) are licensed under the MIT License. You may freely use, modify, and distribute them in accordance with the MIT License terms.

- Images from the Waterloo Exploration Database (University of Waterloo) are subject to the original restricted license: **non-commercial research and educational purposes only**. Commercial use is strictly prohibited. Full terms and conditions: https://kedema.org/project/exploration/index.html

List of the images from the Waterloo Exploration Database:

- pristine_images___00329
- pristine_images___00530
- pristine_images___01138
- pristine_images___01409
- pristine_images___03427
- pristine_images___03736
- pristine_images___04323