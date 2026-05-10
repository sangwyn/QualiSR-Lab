## Quickstart With Precomputed Features

The fastest smoke test uses the precomputed CSV files already tracked in this
repository.

```bash
python -m pip install -e ".[regressors]"
qualisr-run-regressors --config configs/default.json
```

Or build Docker image:

```bash
docker build -t qualisr-lab .
docker run --rm qualisr-lab
```

You can run any of the following commands inside the Docker container:

```bash
docker run --rm -it --mount type=bind,source="${PWD}",target=/workspace qualisr-lab bash
qualisr --help
qualisr-run-regressors --config configs/default.json

```


## Installation Options


For the regression pipeline only:

```bash
python -m pip install -e ".[regressors]"
```

For full feature extraction on CPU:

```bash
python -m pip install -e ".[features,regressors]"
```

For development:

```bash
python -m pip install -e ".[dev,regressors]"
pytest
```

The legacy fully pinned environment is kept in `requirements.txt`.

## Dataset Layout

The scripts expect this directory structure after downloading or preparing data:

```text
dataset/
  hr/
    0000001.png
  lr/
    0000001.png
  sr/
    PASD/0000001.png
    SUPIR/0000001.png
    RealESRGAN/0000001.png
  ref/
    bicubic/0000001@PASD@bicubic.png
    rlfn/0000001@PASD@rlfn.png
    span/0000001@PASD@span.png
  heatmaps/
    PASD/0000001.npy.gz
```

Quality labels for the sample dataset are stored in `scores/labels.csv`:

```csv
test_case,method,score_norm
0000001,pasd,0.72
0000001,supir,0.25
```

See [dataset/readme.md](dataset/readme.md) for dataset download notes and
publication-readiness metadata that still needs archival documentation.

## Command-Line Workflow

All major scripts are available as console commands.

### 1. Generate Quasi-References

```bash
qualisr-make-reference \
  --lr-dir dataset/lr \
  --sr-dirs PASD=dataset/sr/PASD SUPIR=dataset/sr/SUPIR RealESRGAN=dataset/sr/RealESRGAN \
  --out-root dataset/ref \
  --refs bicubic rlfn span \
  --scale 4 \
  --rlfn-script realtime_sr/RLFN/inference-RLFN.py \
  --rlfn-ckpt realtime_sr/RLFN/rlfn-tuned-4x.pth \
  --span-script realtime_sr/SPAN/inference-SPAN.py \
  --span-ckpt realtime_sr/SPAN/span-tuned-4x.pth
```

### 2. Extract Image Features

```bash
qualisr-extract-features \
  --sr-dirs PASD=dataset/sr/PASD SUPIR=dataset/sr/SUPIR RealESRGAN=dataset/sr/RealESRGAN \
  --gt-dir dataset/hr \
  --lr-dir dataset/lr \
  --ref-dirs bicubic=dataset/ref/bicubic rlfn=dataset/ref/rlfn span=dataset/ref/span \
  --features fr,nr,vgg,resnet,siglip \
  --output features/image_features.csv \
  --device cuda
```

The repository also includes precomputed feature blocks such as
`features/nr@grounding.csv`, `features/fr@grounding.csv`,
`features/siglip@grounding.csv`, and PCA-reduced encoder features under
`features/pca/`.

### 3. Apply PCA

```bash
qualisr-apply-pca \
  --input features/image_features.csv \
  --blocks vgg=vgg_ resnet=resnet_ \
  --n-components 5 10 25 50 75 \
  --test-size 0.2 \
  --split-seed 42 \
  --output-dir features/pca
```

### 4. Compute Artifact-Mask Statistics

```bash
qualisr-compute-stats \
  --heatmap-dirs PASD=dataset/heatmaps/PASD SUPIR=dataset/heatmaps/SUPIR RealESRGAN=dataset/heatmaps/RealESRGAN \
  --output features/stats@grounding.csv \
  --percentiles 5 95 \
  --area-thresholds 0 0.5 0.75
```

Artifact masks are treated as an optional input interface. If masks are a central
paper contribution, document or release the mask-generation method before
submission.

### 5. Run Regressors

```bash
qualisr-run-regressors --config configs/default.json
```

