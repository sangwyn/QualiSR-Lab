import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF


FR_METRICS: Sequence[str] = (
    "psnr",
    "ssim",
    "lpips-vgg",
    "stlpips-vgg",
    "pieapp",
    "ahiq",
)

NR_METRICS: Sequence[str] = (
    "musiq",
    "arniqa",
    "qalign",
    "unique",
    "paq2piq",
)

NOISE_COMPONENTS = 5
NOISE_SEED = 42

DEFAULT_FEATURES: Sequence[str] = ("fr", "nr", "vgg", "resnet", "siglip", "gaussian", "uniform")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


LOGGER = logging.getLogger("get_image_features")


class DirectoryIndex:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.files: List[Path] = []
        self.by_name: Dict[str, Path] = {}
        self.by_stem: Dict[str, List[Path]] = defaultdict(list)
        self._index()

    def _index(self) -> None:
        for path in sorted(self.directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            self.files.append(path)
            self.by_name[path.name.lower()] = path
            self.by_stem[path.stem.lower()].append(path)

        for stem in self.by_stem:
            self.by_stem[stem].sort(key=lambda p: p.name.lower())

    def find_same_name(self, source: Path) -> Optional[Path]:
        exact = self.by_name.get(source.name.lower())
        if exact is not None:
            return exact

        return self._pick_candidate(self.by_stem.get(source.stem.lower(), []), source.suffix.lower())

    def find_ref_name(self, source: Path, sr_method: str, ref_name: str) -> Optional[Path]:
        expected_stem = f"{source.stem}@{sr_method}@{ref_name}".lower()
        direct = self._pick_candidate(self.by_stem.get(expected_stem, []), source.suffix.lower())
        if direct is not None:
            return direct

        suffix = f"@{sr_method}@{ref_name}".lower()
        source_stem = source.stem.lower()
        candidates: List[Path] = []
        for stem_key, paths in self.by_stem.items():
            if stem_key.startswith(source_stem) and stem_key.endswith(suffix):
                candidates.extend(paths)

        return self._pick_candidate(candidates, source.suffix.lower())

    @staticmethod
    def _pick_candidate(candidates: Sequence[Path], preferred_ext: str) -> Optional[Path]:
        if not candidates:
            return None

        for candidate in candidates:
            if candidate.suffix.lower() == preferred_ext:
                return candidate

        return sorted(candidates, key=lambda p: p.name.lower())[0]


def parse_named_directories(specs: Iterable[str], flag_name: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}

    for spec in specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
            name = name.strip()
            raw_path = raw_path.strip()
        else:
            raw_path = spec.strip()
            name = Path(raw_path).name

        if not name:
            raise ValueError(f"Invalid {flag_name} value '{spec}': missing name before '='")

        if not raw_path:
            raise ValueError(f"Invalid {flag_name} value '{spec}': missing directory path")

        path = Path(raw_path).expanduser().resolve()
        if name in parsed:
            raise ValueError(f"Duplicate {flag_name} name '{name}'")

        parsed[name] = path

    return parsed


def require_existing_directories(named_dirs: Dict[str, Path], flag_name: str) -> None:
    for name, directory in named_dirs.items():
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"{flag_name} '{name}' points to missing directory: {directory}")


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(image).unsqueeze(0).to(device, dtype=torch.float32)


def center_crop_to_size(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    if target_width > width or target_height > height:
        raise ValueError(
            f"Cannot crop image of size {width}x{height} to larger size {target_width}x{target_height}"
        )

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return image.crop((left, top, right, bottom))


def align_fr_images(
    sr_image: Image.Image,
    ref_image: Image.Image,
    sr_path: Path,
    ref_path: Path,
    ref_name: str,
    strict: bool,
) -> Tuple[Image.Image, Image.Image]:
    sr_width, sr_height = sr_image.size
    ref_width, ref_height = ref_image.size

    if (sr_width, sr_height) == (ref_width, ref_height):
        return sr_image, ref_image

    if ref_width >= sr_width and ref_height >= sr_height:
        cropped_ref = center_crop_to_size(ref_image, sr_width, sr_height)
        LOGGER.warning(
            "Cropped FR reference '%s' from %dx%d to %dx%d for %s",
            ref_name,
            ref_width,
            ref_height,
            sr_width,
            sr_height,
            sr_path.name,
        )
        return sr_image, cropped_ref

    common_width = min(sr_width, ref_width)
    common_height = min(sr_height, ref_height)
    message = (
        f"FR size mismatch for {sr_path.name} vs {ref_path.name}: "
        f"SR={sr_width}x{sr_height}, reference={ref_width}x{ref_height}. "
        f"Reference is smaller than SR, so both images will be center-cropped to "
        f"{common_width}x{common_height}."
    )
    if strict:
        raise RuntimeError(message)
    LOGGER.warning(message)
    cropped_sr = center_crop_to_size(sr_image, common_width, common_height)
    cropped_ref = center_crop_to_size(ref_image, common_width, common_height)
    return cropped_sr, cropped_ref


def init_fr_models(device: torch.device) -> Dict[str, object]:
    import pyiqa

    models_dict = {}
    for metric in FR_METRICS:
        LOGGER.info("Initializing FR metric: %s", metric)
        models_dict[metric] = pyiqa.create_metric(metric, device=device)
    return models_dict


def init_nr_models(device: torch.device) -> Dict[str, object]:
    import pyiqa

    models_dict = {}
    for metric in NR_METRICS:
        LOGGER.info("Initializing NR metric: %s", metric)
        models_dict[metric] = pyiqa.create_metric(metric, device=device)
    return models_dict


def init_vgg(device: torch.device) -> Tuple[torch.nn.Module, transforms.Compose]:
    LOGGER.info("Initializing VGG16 backbone")
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights).features.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def init_resnet(device: torch.device) -> Tuple[torch.nn.Module, transforms.Compose]:
    LOGGER.info("Initializing ResNet50 backbone")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def init_siglip(model_name: str, device: torch.device) -> Tuple[object, object]:
    from transformers import AutoModel, AutoProcessor

    LOGGER.info("Initializing SigLIP model: %s", model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def extract_pretrained_features(
    image: Image.Image,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0).to(device)
        features = model(tensor).flatten().detach().cpu().numpy().astype(np.float32)
    return features


def siglip_embedding(image: Image.Image, model: object, processor: object, device: torch.device) -> torch.Tensor:
    inputs = processor(text=[""], images=image, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.image_embeds
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)

    return F.normalize(embedding, dim=-1)


def compute_siglip_scores(
    lr_image: Image.Image,
    sr_image: Image.Image,
    model: object,
    processor: object,
    device: torch.device,
    alpha: float,
) -> Dict[str, float]:
    lr_upscaled = lr_image.resize(sr_image.size, Image.Resampling.BICUBIC)

    lr_feat = siglip_embedding(lr_image, model, processor, device)
    lr_up_feat = siglip_embedding(lr_upscaled, model, processor, device)
    sr_feat = siglip_embedding(sr_image, model, processor, device)

    content_fidelity = F.cosine_similarity(sr_feat, lr_up_feat).item()
    perceptual_enhancement = F.cosine_similarity(sr_feat, lr_feat).item()
    final_rr_score = alpha * content_fidelity + (1.0 - alpha) * perceptual_enhancement

    return {
        "content_fidelity": float(content_fidelity),
        "perceptual_enhancement": float(perceptual_enhancement),
        "final_rr_score": float(final_rr_score),
    }


def ensure_tensor(image_tensor: Optional[torch.Tensor], image_path: Optional[Path], device: torch.device) -> Optional[torch.Tensor]:
    if image_tensor is not None or image_path is None:
        return image_tensor

    image = load_image_rgb(image_path)
    return image_to_tensor(image, device)


def parse_features(raw_features: str) -> List[str]:
    requested = []
    for part in raw_features.split(","):
        feature = part.strip().lower()
        if not feature:
            continue
        if feature not in DEFAULT_FEATURES:
            raise ValueError(f"Unknown feature '{feature}'. Supported: {', '.join(DEFAULT_FEATURES)}")
        requested.append(feature)

    if not requested:
        raise ValueError("At least one feature must be requested.")

    deduped = list(dict.fromkeys(requested))
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FR/NR/VGG/ResNet/SigLIP features for SR images and save one CSV. "
            "SR methods are passed as METHOD=DIR. "
            "Reference image names are expected as <sr_stem>@<sr_method>@<ref_name>.<ext>."
        )
    )

    parser.add_argument(
        "--sr-dirs",
        nargs="+",
        required=True,
        metavar="METHOD=DIR",
        help="One or more SR folders. Example: --sr-dirs SwinIR=/data/sr/swinir ATD=/data/sr/atd",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="GT image directory. File names are matched to SR by same name/stem.",
    )
    parser.add_argument(
        "--lr-dir",
        type=str,
        default=None,
        help="LR image directory. Required for SigLIP. File names are matched to SR by same name/stem.",
    )
    parser.add_argument(
        "--ref-dirs",
        nargs="*",
        default=(),
        metavar="NAME=DIR",
        help=(
            "Additional FR references (bicubic/RLFN/SPAN/etc). "
            "Each file is expected as <sr_stem>@<sr_method>@<ref_name>.<ext>."
        ),
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated subset of features: fr,nr,vgg,resnet,siglip,gaussian,uniform.",
    )
    parser.add_argument(
        "--siglip-model",
        default="google/siglip2-base-patch16-256",
        help="Hugging Face model id for SigLIP feature extraction.",
    )
    parser.add_argument(
        "--siglip-alpha",
        type=float,
        default=0.5,
        help="SigLIP final score weight (alpha * content + (1-alpha) * enhancement).",
    )
    parser.add_argument(
        "--noise-components",
        type=int,
        default=NOISE_COMPONENTS,
        help="Number of noise features (if gaussian/uniform specified in feature list).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=NOISE_SEED,
        help="Random seed for noise features (if gaussian/uniform specified in feature list).",
    )
    parser.add_argument("--output", required=True, help="Output CSV file path.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing matches or metric errors instead of writing NaNs and continuing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )

    return parser.parse_args()


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "cpu":
        return torch.device("cpu")
    if raw_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_raise_or_warn(message: str, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    LOGGER.warning(message)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    requested_features = parse_features(args.features)
    device = resolve_device(args.device)

    sr_dirs = parse_named_directories(args.sr_dirs, "--sr-dirs")
    ref_dirs = parse_named_directories(args.ref_dirs, "--ref-dirs")

    require_existing_directories(sr_dirs, "--sr-dirs")
    require_existing_directories(ref_dirs, "--ref-dirs")

    gt_index: Optional[DirectoryIndex] = None
    if args.gt_dir is not None:
        gt_path = Path(args.gt_dir).expanduser().resolve()
        if not gt_path.exists() or not gt_path.is_dir():
            raise FileNotFoundError(f"--gt-dir points to missing directory: {gt_path}")
        gt_index = DirectoryIndex(gt_path)

    lr_index: Optional[DirectoryIndex] = None
    if args.lr_dir is not None:
        lr_path = Path(args.lr_dir).expanduser().resolve()
        if not lr_path.exists() or not lr_path.is_dir():
            raise FileNotFoundError(f"--lr-dir points to missing directory: {lr_path}")
        lr_index = DirectoryIndex(lr_path)

    NOISE_COMPONENTS = args.noise_components
    NOISE_SEED = args.noise_seed

    if "siglip" in requested_features and lr_index is None:
        raise ValueError("SigLIP requires --lr-dir")

    if "fr" in requested_features and gt_index is None and not ref_dirs:
        raise ValueError("FR metrics require at least one reference directory: --gt-dir and/or --ref-dirs")

    sr_indices = {name: DirectoryIndex(directory) for name, directory in sr_dirs.items()}
    ref_indices = {name: DirectoryIndex(directory) for name, directory in ref_dirs.items()}

    total_sr_images = sum(len(index.files) for index in sr_indices.values())
    LOGGER.info("Found %d SR images across %d methods", total_sr_images, len(sr_indices))

    fr_models: Optional[Dict[str, object]] = None
    if "fr" in requested_features:
        fr_models = init_fr_models(device)

    nr_models: Optional[Dict[str, object]] = None
    if "nr" in requested_features:
        nr_models = init_nr_models(device)

    vgg_model: Optional[torch.nn.Module] = None
    vgg_transform: Optional[transforms.Compose] = None
    if "vgg" in requested_features:
        vgg_model, vgg_transform = init_vgg(device)

    resnet_model: Optional[torch.nn.Module] = None
    resnet_transform: Optional[transforms.Compose] = None
    if "resnet" in requested_features:
        resnet_model, resnet_transform = init_resnet(device)

    siglip_model = None
    siglip_processor = None
    if "siglip" in requested_features:
        siglip_model, siglip_processor = init_siglip(args.siglip_model, device)

    rows: List[Dict[str, float]] = []
    processed = 0

    for sr_method, sr_index in sr_indices.items():
        LOGGER.info("Processing SR method '%s' (%d images)", sr_method, len(sr_index.files))

        for sr_path in sr_index.files:
            processed += 1
            LOGGER.info("[%d/%d] %s/%s", processed, total_sr_images, sr_method, sr_path.name)

            try:
                sr_image = load_image_rgb(sr_path)
            except Exception as exc:
                maybe_raise_or_warn(f"Failed to load SR image {sr_path}: {exc}", args.strict)
                continue

            sr_tensor = image_to_tensor(sr_image, device)

            row: Dict[str, float] = {
                "sample_id": f"{sr_method}:{sr_path.stem}",
                "sr_method": sr_method,
                "sr_filename": sr_path.name,
                "sr_path": str(sr_path),
            }

            gt_path: Optional[Path] = None
            if gt_index is not None:
                gt_path = gt_index.find_same_name(sr_path)
                row["gt_path"] = str(gt_path) if gt_path is not None else ""

            lr_path: Optional[Path] = None
            if lr_index is not None:
                lr_path = lr_index.find_same_name(sr_path)
                row["lr_path"] = str(lr_path) if lr_path is not None else ""

            if "nr" in requested_features and nr_models is not None:
                for metric_name, model in nr_models.items():
                    try:
                        with torch.no_grad():
                            row[metric_name] = float(model(sr_tensor).item())
                    except Exception as exc:
                        maybe_raise_or_warn(
                            f"NR metric '{metric_name}' failed on {sr_path.name}: {exc}",
                            args.strict,
                        )
                        row[metric_name] = np.nan

            if "fr" in requested_features and fr_models is not None:
                fr_targets: List[Tuple[str, Optional[Path], str]] = []
                if gt_index is not None:
                    fr_targets.append(("gt", gt_path, "same-name"))

                for ref_name, ref_index in ref_indices.items():
                    ref_path = ref_index.find_ref_name(sr_path, sr_method=sr_method, ref_name=ref_name)
                    fr_targets.append((ref_name, ref_path, "suffix"))

                for ref_name, ref_path, match_mode in fr_targets:
                    if ref_path is None:
                        maybe_raise_or_warn(
                            f"Missing FR reference '{ref_name}' for {sr_path.name} ({match_mode} match)",
                            args.strict,
                        )
                        for metric_name in FR_METRICS:
                            row[f"{metric_name}_{ref_name}"] = np.nan
                        continue

                    try:
                        ref_image = load_image_rgb(ref_path)
                        sr_image_fr, ref_image_fr = align_fr_images(
                            sr_image=sr_image,
                            ref_image=ref_image,
                            sr_path=sr_path,
                            ref_path=ref_path,
                            ref_name=ref_name,
                            strict=args.strict,
                        )
                        if sr_image_fr.size == sr_image.size:
                            sr_tensor_fr = sr_tensor
                        else:
                            sr_tensor_fr = image_to_tensor(sr_image_fr, device)
                        ref_tensor = image_to_tensor(ref_image_fr, device)
                    except Exception as exc:
                        maybe_raise_or_warn(
                            f"Failed to prepare FR pair ({sr_path.name}, {ref_path.name}): {exc}",
                            args.strict,
                        )
                        for metric_name in FR_METRICS:
                            row[f"{metric_name}_{ref_name}"] = np.nan
                        continue

                    for metric_name, model in fr_models.items():
                        out_col = f"{metric_name}_{ref_name}"
                        try:
                            with torch.no_grad():
                                row[out_col] = float(model(sr_tensor_fr, ref_tensor).item())
                        except Exception as exc:
                            maybe_raise_or_warn(
                                f"FR metric '{metric_name}' failed on {sr_path.name} vs {ref_path.name}: {exc}",
                                args.strict,
                            )
                            row[out_col] = np.nan
                
            if "vgg" in requested_features and vgg_model is not None and vgg_transform is not None:
                try:
                    vgg_features = extract_pretrained_features(sr_image, vgg_model, vgg_transform, device)
                    for index, value in enumerate(vgg_features):
                        row[f"vgg_{index:05d}"] = float(value)
                except Exception as exc:
                    maybe_raise_or_warn(f"VGG extraction failed on {sr_path.name}: {exc}", args.strict)

            if "resnet" in requested_features and resnet_model is not None and resnet_transform is not None:
                try:
                    resnet_features = extract_pretrained_features(sr_image, resnet_model, resnet_transform, device)
                    for index, value in enumerate(resnet_features):
                        row[f"resnet_{index:05d}"] = float(value)
                except Exception as exc:
                    maybe_raise_or_warn(f"ResNet extraction failed on {sr_path.name}: {exc}", args.strict)

            if "siglip" in requested_features and siglip_model is not None and siglip_processor is not None:
                if lr_path is None:
                    maybe_raise_or_warn(f"Missing LR image for SigLIP: {sr_path.name}", args.strict)
                    row["content_fidelity"] = np.nan
                    row["perceptual_enhancement"] = np.nan
                    row["final_rr_score"] = np.nan
                else:
                    try:
                        lr_image = load_image_rgb(lr_path)
                        siglip_scores = compute_siglip_scores(
                            lr_image=lr_image,
                            sr_image=sr_image,
                            model=siglip_model,
                            processor=siglip_processor,
                            device=device,
                            alpha=args.siglip_alpha,
                        )
                        row.update(siglip_scores)
                    except Exception as exc:
                        maybe_raise_or_warn(f"SigLIP extraction failed on {sr_path.name}: {exc}", args.strict)
                        row["content_fidelity"] = np.nan
                        row["perceptual_enhancement"] = np.nan
                        row["final_rr_score"] = np.nan

            rows.append(row)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)

    if "gaussian" in requested_features:
        try:
            np.random.seed(NOISE_SEED)
            sample = np.random.normal(size=(len(rows), NOISE_COMPONENTS))
            sample = pd.DataFrame(sample, columns=[f"gaussian_{i}" for i in range(NOISE_COMPONENTS)])
            frame = pd.concat([frame, sample], axis=1)
        except Exception as exc:
            maybe_raise_or_warn(f"Gaussian noise sampling failed: {exc}", args.strict)

    if "uniform" in requested_features:
        try:
            np.random.seed(NOISE_SEED)
            sample = np.random.uniform(size=(len(rows), NOISE_COMPONENTS))
            sample = pd.DataFrame(sample, columns=[f"uniform_{i}" for i in range(NOISE_COMPONENTS)])
            frame = pd.concat([frame, sample], axis=1)
        except Exception as exc:
            maybe_raise_or_warn(f"Uniform noise sampling failed: {exc}", args.strict)

    frame.to_csv(output_path, index=False)
    LOGGER.info("Saved %d rows to %s", len(frame), output_path)


if __name__ == "__main__":
    main()
