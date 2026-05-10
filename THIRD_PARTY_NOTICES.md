# Third-Party Notices

This repository combines original QualiSR-Lab code with interfaces to external
research models, metrics, and data sources. The root `LICENSE` applies to the
original QualiSR-Lab code unless a file or directory states otherwise.

## Included Real-Time Super-Resolution Assets

The `realtime_sr/` directory contains inference code and checkpoint files used
to generate quasi-reference images:

- `realtime_sr/RLFN/` references Residual Local Feature Network (RLFN):
  https://github.com/bytedance/RLFN
- `realtime_sr/SPAN/` references Swift Parameter-free Attention Network (SPAN):
  https://github.com/zononhzy/SPAN

Before public archival release, verify that the upstream licenses permit
redistribution of both the code and the `.pth` checkpoint files in this
repository. If redistribution is not permitted, replace the checkpoints with
download scripts and checksums.

## External Metrics And Models

Feature extraction can call third-party libraries and model weights, including:

- PyIQA and its supported NR/FR metrics
- VGG16 and ResNet50 ImageNet weights from torchvision
- SigLIP models from Hugging Face Transformers
- XGBoost and CatBoost for regression experiments

Users are responsible for following the licenses of downloaded model weights,
datasets, and package dependencies.
