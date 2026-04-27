# Grounding dataset

## 🔎 Overview

This is a sample dataset based on [Q-Instruct](https://q-future.github.io/Q-Instruct/) training set. We picked 40 ground-truth images based on K-Means clusterization (8 images for each of the 5 centroids). For each of these images, we provide both SR ([PASD](https://github.com/yangxy/PASD)/[SUPIR](https://github.com/Fanghua-Yu/SUPIR)/[RealESRGAN](https://github.com/xinntao/real-esrgan)) and reference ([RLFN](https://github.com/bytedance/RLFN)/[SPAN](https://github.com/zononhzy/SPAN)/bicubic) images and also artifact masks in `heatmaps/` subdirectory. Each SR image has a Mean Opinion Score assigned to it in `labels.csv`.

## ⚙️ How to use

Dataset is available to download for free on [GDrive](https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing) or [GML server](https://titan.gml-team.ru:5003/sharing/hm3FpzpDp). Below are the commands to download and unpack all the images.

```bash
gdown 'https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing' --fuzzy
unzip grounding_dataset.zip -d dataset/
```
