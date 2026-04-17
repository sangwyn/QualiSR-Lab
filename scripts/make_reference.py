"""Generate reference images (bicubic / RLFN / SPAN) for SR-IQA datasets.

Output naming:
    <sr_stem>@<sr_method>@<ref_suffix><output_ext>

Typical usage:
    python scripts/make_reference.py \
      --lr-dir /data/LR \
      --sr-dirs PASD=/data/SR/PASD SUPIR=/data/SR/SUPIR RealESRGAN=/data/SR/RealESRGAN \
      --out-root /data/references \
      --refs bicubic rlfn span \
      --scale 4 \
      --rlfn-script ./realtime_sr/RLFN/inference-RLFN.py --rlfn-ckpt ./realtime_sr/RLFN/rlfn-tuned-4x.pth \
      --span-script ./realtime_sr/SPAN/inference-SPAN.py --span-ckpt ./realtime_sr/SPAN/span-tuned-4x.pth
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm


LOGGER = logging.getLogger("gt_rf")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_REFS = ("bicubic", "rlfn", "span")

DEFAULT_RLFN_TEMPLATE = (
    "{python} {script} --ckpt {ckpt} --scale {scale} --image {input} --output {output}"
)
DEFAULT_SPAN_TEMPLATE = (
    "{python} {script} --ckpt {ckpt} --scale {scale} --image {input} --output {output}"
)


class ImageIndex:
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

    def find_match(self, source: Path) -> Optional[Path]:
        exact = self.by_name.get(source.name.lower())
        if exact is not None:
            return exact

        candidates = self.by_stem.get(source.stem.lower(), [])
        if not candidates:
            return None

        preferred_ext = source.suffix.lower()
        for candidate in candidates:
            if candidate.suffix.lower() == preferred_ext:
                return candidate

        return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference images for SR outputs (bicubic / RLFN / SPAN).",
    )

    parser.add_argument("--lr-dir", required=True, help="Directory with LR images.")
    parser.add_argument(
        "--sr-dirs",
        nargs="+",
        required=True,
        metavar="METHOD=DIR",
        help="SR method directories. Example: PASD=/data/pasd SUPIR=/data/supir",
    )
    parser.add_argument(
        "--refs",
        nargs="+",
        default=list(SUPPORTED_REFS),
        choices=SUPPORTED_REFS,
        help="Which references to generate.",
    )
    parser.add_argument(
        "--ref-dirs",
        nargs="*",
        default=(),
        metavar="REF=DIR",
        help="Optional explicit output dirs per reference. REF in {bicubic,rlfn,span}",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root for references. Used for refs not provided via --ref-dirs.",
    )

    parser.add_argument("--output-ext", default=".png", help="Output extension (default: .png)")
    parser.add_argument("--bicubic-suffix", default="bicubic", help="Suffix for bicubic filenames")
    parser.add_argument("--rlfn-suffix", default="rlfn", help="Suffix for RLFN filenames")
    parser.add_argument("--span-suffix", default="span", help="Suffix for SPAN filenames")

    parser.add_argument("--scale", type=int, default=4, help="Scale argument passed to SR model inference")
    parser.add_argument("--python-exec", default="python3", help="Python executable for inference commands")

    parser.add_argument("--rlfn-script", default='./realtime_sr/RLFN/inference-RLFN.py', help="Path to RLFN inference script")
    parser.add_argument("--rlfn-ckpt", default='./realtime_sr/RLFN/rlfn-tuned-4x.pth', help="Path to RLFN checkpoint")
    parser.add_argument(
        "--rlfn-cmd-template",
        default=DEFAULT_RLFN_TEMPLATE,
        help=(
            "Template for RLFN command with placeholders: "
            "{python}, {script}, {ckpt}, {scale}, {input}, {output}"
        ),
    )

    parser.add_argument("--span-script", default='./realtime_sr/SPAN/inference-SPAN.py', help="Path to SPAN inference script")
    parser.add_argument("--span-ckpt", default='./realtime_sr/SPAN/span-tuned-4x.pth', help="Path to SPAN checkpoint")
    parser.add_argument(
        "--span-cmd-template",
        default=DEFAULT_SPAN_TEMPLATE,
        help=(
            "Template for SPAN command with placeholders: "
            "{python}, {script}, {ckpt}, {scale}, {input}, {output}"
        ),
    )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing reference files")
    parser.add_argument("--strict", action="store_true", help="Stop on first error")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N SR images per method")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity",
    )

    return parser.parse_args()


def parse_named_paths(
    specs: Iterable[str],
    flag_name: str,
    lowercase_keys: bool = False,
) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}

    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{flag_name} expects NAME=PATH entries. Got: '{spec}'")

        name, raw_path = spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()

        if not name or not raw_path:
            raise ValueError(f"Invalid {flag_name} value: '{spec}'")

        key = name.lower() if lowercase_keys else name
        if key in parsed:
            raise ValueError(f"Duplicate {flag_name} key '{name}'")

        parsed[key] = Path(raw_path).expanduser().resolve()

    return parsed


def ensure_existing_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def resolve_output_dirs(args: argparse.Namespace, refs: Sequence[str]) -> Dict[str, Path]:
    ref_dirs = (
        parse_named_paths(args.ref_dirs, "--ref-dirs", lowercase_keys=True) if args.ref_dirs else {}
    )

    unknown = [name for name in ref_dirs if name not in SUPPORTED_REFS]
    if unknown:
        raise ValueError(f"Unsupported refs in --ref-dirs: {unknown}. Supported: {SUPPORTED_REFS}")

    if args.out_root is None and any(ref not in ref_dirs for ref in refs):
        missing = [ref for ref in refs if ref not in ref_dirs]
        raise ValueError(
            f"Missing output directories for refs {missing}. Provide --out-root or explicit --ref-dirs."
        )

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    resolved: Dict[str, Path] = {}
    for ref in refs:
        if ref in ref_dirs:
            out_dir = ref_dirs[ref]
        else:
            out_dir = out_root / ref
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved[ref] = out_dir

    return resolved


def normalize_output_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        raise ValueError("--output-ext cannot be empty")
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def maybe_raise(message: str, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    LOGGER.warning(message)


def render_command(template: str, values: Dict[str, str]) -> str:
    safe = {k: shlex.quote(v) for k, v in values.items()}
    return template.format(**safe)


def run_inference_command(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def ensure_output_size(path: Path, target_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.size == target_size:
            return
        resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    resized.save(path)


def generate_bicubic(lr_path: Path, sr_size: tuple[int, int], output_path: Path) -> None:
    with Image.open(lr_path) as image:
        bicubic = image.convert("RGB").resize(sr_size, Image.Resampling.BICUBIC)
    bicubic.save(output_path)


def validate_model_requirements(args: argparse.Namespace, refs: Sequence[str]) -> None:
    if "rlfn" in refs:
        if args.rlfn_script is None or args.rlfn_ckpt is None:
            raise ValueError("RLFN requested but --rlfn-script/--rlfn-ckpt are missing")
        if "{input}" not in args.rlfn_cmd_template or "{output}" not in args.rlfn_cmd_template:
            raise ValueError("--rlfn-cmd-template must include {input} and {output}")
        rlfn_script = Path(args.rlfn_script).expanduser().resolve()
        rlfn_ckpt = Path(args.rlfn_ckpt).expanduser().resolve()
        if not rlfn_script.exists():
            raise FileNotFoundError(f"RLFN script not found: {rlfn_script}")
        if not rlfn_ckpt.exists():
            raise FileNotFoundError(f"RLFN checkpoint not found: {rlfn_ckpt}")

    if "span" in refs:
        if args.span_script is None or args.span_ckpt is None:
            raise ValueError("SPAN requested but --span-script/--span-ckpt are missing")
        if "{input}" not in args.span_cmd_template or "{output}" not in args.span_cmd_template:
            raise ValueError("--span-cmd-template must include {input} and {output}")
        span_script = Path(args.span_script).expanduser().resolve()
        span_ckpt = Path(args.span_ckpt).expanduser().resolve()
        if not span_script.exists():
            raise FileNotFoundError(f"SPAN script not found: {span_script}")
        if not span_ckpt.exists():
            raise FileNotFoundError(f"SPAN checkpoint not found: {span_ckpt}")


def build_output_path(
    out_dir: Path,
    sr_stem: str,
    sr_method: str,
    ref_suffix: str,
    output_ext: str,
) -> Path:
    return out_dir / f"{sr_stem}@{sr_method}@{ref_suffix}{output_ext}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    refs = [ref.lower() for ref in args.refs]
    validate_model_requirements(args, refs)

    output_ext = normalize_output_ext(args.output_ext)
    suffix_by_ref = {
        "bicubic": args.bicubic_suffix,
        "rlfn": args.rlfn_suffix,
        "span": args.span_suffix,
    }

    lr_dir = Path(args.lr_dir).expanduser().resolve()
    ensure_existing_dir(lr_dir, "--lr-dir")

    sr_dirs = parse_named_paths(args.sr_dirs, "--sr-dirs", lowercase_keys=False)
    if not sr_dirs:
        raise ValueError("--sr-dirs is empty")
    for method, directory in sr_dirs.items():
        ensure_existing_dir(directory, f"SR dir ({method})")

    out_dirs = resolve_output_dirs(args, refs)

    lr_index = ImageIndex(lr_dir)
    if not lr_index.files:
        raise ValueError(f"No image files found in LR directory: {lr_dir}")

    stats = Counter()

    for sr_method, sr_dir in sr_dirs.items():
        sr_index = ImageIndex(sr_dir)
        sr_files = sr_index.files[: args.limit] if args.limit is not None else sr_index.files

        LOGGER.info("SR method '%s': %d images in %s", sr_method, len(sr_files), sr_dir)
        progress = tqdm(sr_files, desc=f"{sr_method}", unit="img", disable=args.no_progress)

        for sr_path in progress:
            stats["sr_total"] += 1

            lr_path = lr_index.find_match(sr_path)
            if lr_path is None:
                stats["missing_lr"] += 1
                maybe_raise(f"No LR match for SR image: {sr_path}", args.strict)
                continue

            try:
                with Image.open(sr_path) as sr_image:
                    sr_size = sr_image.size
            except Exception as exc:
                stats["failed_sr_load"] += 1
                maybe_raise(f"Failed to load SR image {sr_path}: {exc}", args.strict)
                continue

            if "bicubic" in refs:
                out_path = build_output_path(
                    out_dir=out_dirs["bicubic"],
                    sr_stem=sr_path.stem,
                    sr_method=sr_method,
                    ref_suffix=suffix_by_ref["bicubic"],
                    output_ext=output_ext,
                )
                if out_path.exists() and not args.overwrite:
                    stats["bicubic_skipped_existing"] += 1
                else:
                    try:
                        generate_bicubic(lr_path, sr_size, out_path)
                        stats["bicubic_ok"] += 1
                    except Exception as exc:
                        stats["bicubic_failed"] += 1
                        maybe_raise(f"Bicubic failed for {sr_path.name}: {exc}", args.strict)

            if "rlfn" in refs:
                out_path = build_output_path(
                    out_dir=out_dirs["rlfn"],
                    sr_stem=sr_path.stem,
                    sr_method=sr_method,
                    ref_suffix=suffix_by_ref["rlfn"],
                    output_ext=output_ext,
                )
                if out_path.exists() and not args.overwrite:
                    stats["rlfn_skipped_existing"] += 1
                else:
                    cmd = render_command(
                        args.rlfn_cmd_template,
                        {
                            "python": args.python_exec,
                            "script": str(Path(args.rlfn_script).expanduser().resolve()),
                            "ckpt": str(Path(args.rlfn_ckpt).expanduser().resolve()),
                            "scale": str(args.scale),
                            "input": str(lr_path),
                            "output": str(out_path),
                        },
                    )
                    result = run_inference_command(cmd)
                    if result.returncode != 0 or not out_path.exists():
                        stats["rlfn_failed"] += 1
                        stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
                        maybe_raise(
                            f"RLFN failed for {sr_path.name}. Exit={result.returncode}. "
                            f"Stderr tail: {' | '.join(stderr_tail)}",
                            args.strict,
                        )
                    else:
                        try:
                            ensure_output_size(out_path, sr_size)
                            stats["rlfn_ok"] += 1
                        except Exception as exc:
                            stats["rlfn_failed"] += 1
                            maybe_raise(
                                f"RLFN output resize failed for {sr_path.name}: {exc}",
                                args.strict,
                            )

            if "span" in refs:
                out_path = build_output_path(
                    out_dir=out_dirs["span"],
                    sr_stem=sr_path.stem,
                    sr_method=sr_method,
                    ref_suffix=suffix_by_ref["span"],
                    output_ext=output_ext,
                )
                if out_path.exists() and not args.overwrite:
                    stats["span_skipped_existing"] += 1
                else:
                    cmd = render_command(
                        args.span_cmd_template,
                        {
                            "python": args.python_exec,
                            "script": str(Path(args.span_script).expanduser().resolve()),
                            "ckpt": str(Path(args.span_ckpt).expanduser().resolve()),
                            "scale": str(args.scale),
                            "input": str(lr_path),
                            "output": str(out_path),
                        },
                    )
                    result = run_inference_command(cmd)
                    if result.returncode != 0 or not out_path.exists():
                        stats["span_failed"] += 1
                        stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
                        maybe_raise(
                            f"SPAN failed for {sr_path.name}. Exit={result.returncode}. "
                            f"Stderr tail: {' | '.join(stderr_tail)}",
                            args.strict,
                        )
                    else:
                        try:
                            ensure_output_size(out_path, sr_size)
                            stats["span_ok"] += 1
                        except Exception as exc:
                            stats["span_failed"] += 1
                            maybe_raise(
                                f"SPAN output resize failed for {sr_path.name}: {exc}",
                                args.strict,
                            )

    LOGGER.info("Done. Summary:")
    for key in sorted(stats):
        LOGGER.info("  %s: %d", key, stats[key])


if __name__ == "__main__":
    main()
