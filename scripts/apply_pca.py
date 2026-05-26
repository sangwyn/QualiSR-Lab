import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

LOGGER = logging.getLogger("apply_pca_image_features")
DEFAULT_BLOCKS: Sequence[str] = ("vgg=vgg_", "resnet=resnet_")
PATH_COLUMNS: Sequence[str] = ("sr_path", "gt_path", "lr_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply PCA to high-dimensional feature blocks (e.g. vgg_*, resnet_*) "
            "in CSV files created by get_image_features.py."
        )
    )

    parser.add_argument("--input", required=True, help="Input CSV from get_image_features.py")
    parser.add_argument(
        "--n-components",
        nargs="+",
        required=True,
        type=int,
        help="One or more PCA dimensions to generate (example: --n-components 5 10 25 50)",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=list(DEFAULT_BLOCKS),
        metavar="NAME=PREFIX",
        help=(
            "Feature blocks to reduce. NAME is used in output column names, "
            "PREFIX selects columns by startswith. Default: vgg=vgg_ resnet=resnet_"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Default: input file directory.",
    )
    parser.add_argument(
        "--output-template",
        default="{stem}_pca{n}.csv",
        help="Output file template. Available fields: {stem}, {n}",
    )
    parser.add_argument(
        "--keep-original-blocks",
        action="store_true",
        help="Keep original block columns (vgg_*/resnet_*) in output.",
    )
    parser.add_argument(
        "--fit-column",
        default=None,
        help="Optional column name for selecting rows to fit PCA.",
    )
    parser.add_argument(
        "--fit-value",
        default=None,
        help="Optional value for --fit-column (example: train). If omitted, fit on all rows.",
    )
    parser.add_argument(
        "--disable-auto-split",
        action="store_true",
        help=(
            "Disable automatic grouped train/test split when --fit-column is not provided. "
            "If set, PCA is fit on all rows (unless --fit-column is used)."
        ),
    )
    parser.add_argument(
        "--split-column",
        default="set_type",
        help="Column name to write automatic split labels into (default: set_type).",
    )
    parser.add_argument(
        "--train-label",
        default="train",
        help="Label used for train rows in automatic split.",
    )
    parser.add_argument(
        "--test-label",
        default="test",
        help="Label used for test rows in automatic split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of GT groups assigned to test set in automatic split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for automatic grouped train/test split.",
    )
    parser.add_argument(
        "--group-column",
        default="gt_path",
        help="Primary column for GT grouping in automatic split (default: gt_path).",
    )
    parser.add_argument(
        "--group-fallback-column",
        default="sr_filename",
        help=(
            "Fallback column used when --group-column is missing/empty. "
            "Default uses SR filename stem."
        ),
    )
    parser.add_argument(
        "--svd-solver",
        choices=("auto", "full", "covariance_eigh", "arpack", "randomized"),
        default="full",
        help="scikit-learn PCA solver.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser.parse_args()


def parse_blocks(block_specs: Sequence[str]) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    seen_names = set()

    for spec in block_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --blocks value '{spec}'. Expected NAME=PREFIX.")

        name, prefix = spec.split("=", 1)
        name = name.strip()
        prefix = prefix.strip()

        if not name or not prefix:
            raise ValueError(f"Invalid --blocks value '{spec}'. NAME and PREFIX must be non-empty.")

        if name in seen_names:
            raise ValueError(f"Duplicate block name '{name}' in --blocks.")

        seen_names.add(name)
        blocks.append((name, prefix))

    return blocks


def get_fit_mask(df: pd.DataFrame, fit_column: str, fit_value: str) -> np.ndarray:
    if fit_column not in df.columns:
        raise ValueError(f"--fit-column '{fit_column}' does not exist in input CSV.")

    if fit_value is None:
        raise ValueError("--fit-value is required when --fit-column is used.")

    mask = df[fit_column].astype(str) == str(fit_value)
    if not mask.any():
        raise ValueError(
            f"No rows match fit filter: {fit_column} == {fit_value}. "
            "Cannot fit PCA on empty subset."
        )

    return mask.to_numpy()


def validate_n_components(n_components: Sequence[int]) -> List[int]:
    parsed = sorted(set(n_components))
    if not parsed:
        raise ValueError("No n_components provided.")

    invalid = [n for n in parsed if n <= 0]
    if invalid:
        raise ValueError(f"n_components must be > 0. Invalid values: {invalid}")

    return parsed


def normalize_group_key(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None

    return raw


def fallback_group_key_from_value(value: object) -> Optional[str]:
    normalized = normalize_group_key(value)
    if normalized is None:
        return None

    return Path(normalized).stem


def relativize_path_value(value: object) -> object:
    if pd.isna(value):
        return value

    raw = str(value).strip()
    if not raw:
        return value

    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw

    try:
        return Path(os.path.relpath(path, start=Path.cwd())).as_posix()
    except ValueError:
        return path.as_posix()


def relativize_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in PATH_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(relativize_path_value)
    return df


def build_group_keys(
    df: pd.DataFrame,
    group_column: str,
    group_fallback_column: str,
) -> np.ndarray:
    group_keys: List[Optional[str]] = [None] * len(df)

    if group_column in df.columns:
        for i, value in enumerate(df[group_column]):
            group_keys[i] = normalize_group_key(value)
    else:
        LOGGER.warning(
            "Group column '%s' is missing. Falling back to '%s'.",
            group_column,
            group_fallback_column,
        )

    if group_fallback_column not in df.columns and any(key is None for key in group_keys):
        raise ValueError(
            f"Some rows have no group key from '{group_column}', and fallback column "
            f"'{group_fallback_column}' does not exist."
        )

    if group_fallback_column in df.columns:
        fallback_values = df[group_fallback_column].tolist()
        for i, key in enumerate(group_keys):
            if key is not None:
                continue
            group_keys[i] = fallback_group_key_from_value(fallback_values[i])

    unresolved = [i for i, key in enumerate(group_keys) if key is None]
    if unresolved:
        raise ValueError(
            f"Failed to derive GT group keys for {len(unresolved)} rows. "
            f"Check '{group_column}' or '{group_fallback_column}'."
        )

    return np.asarray(group_keys, dtype=object)


def make_grouped_split(
    df: pd.DataFrame,
    split_column: str,
    group_column: str,
    group_fallback_column: str,
    test_size: float,
    split_seed: int,
    train_label: str,
    test_label: str,
) -> Tuple[np.ndarray, pd.Series, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be in (0, 1). Got {test_size}.")

    group_keys = build_group_keys(df, group_column, group_fallback_column)
    unique_groups = np.unique(group_keys)
    if unique_groups.size < 2:
        raise ValueError(
            "Automatic split requires at least 2 distinct GT groups. "
            f"Found {unique_groups.size}."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
    all_indices = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(all_indices, groups=group_keys))

    labels = np.full(len(df), test_label, dtype=object)
    labels[train_idx] = train_label
    labels_series = pd.Series(labels, index=df.index, name=split_column)

    check = pd.DataFrame({"group": group_keys, "split": labels})
    leaks = check.groupby("group")["split"].nunique()
    leaked_groups = leaks[leaks > 1]
    if not leaked_groups.empty:
        raise RuntimeError(
            "Split leakage detected: some GT groups are assigned to both train and test."
        )

    fit_mask = labels_series == train_label
    return fit_mask.to_numpy(), labels_series, group_keys


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    n_values = validate_n_components(args.n_components)
    max_n = max(n_values)

    blocks = parse_blocks(args.blocks)

    LOGGER.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    fit_mask: np.ndarray
    split_labels: Optional[pd.Series] = None
    if args.fit_column is not None:
        fit_mask = get_fit_mask(df, args.fit_column, args.fit_value)
    elif args.disable_auto_split:
        fit_mask = np.ones(len(df), dtype=bool)
    else:
        fit_mask, split_labels, group_keys = make_grouped_split(
            df=df,
            split_column=args.split_column,
            group_column=args.group_column,
            group_fallback_column=args.group_fallback_column,
            test_size=args.test_size,
            split_seed=args.split_seed,
            train_label=args.train_label,
            test_label=args.test_label,
        )
        df[args.split_column] = split_labels

        train_rows = int((split_labels == args.train_label).sum())
        test_rows = int((split_labels == args.test_label).sum())
        train_groups = int(pd.Series(group_keys[split_labels == args.train_label]).nunique())
        test_groups = int(pd.Series(group_keys[split_labels == args.test_label]).nunique())
        LOGGER.info(
            "Auto split complete (seed=%d): train rows=%d, test rows=%d, train groups=%d, test groups=%d",
            args.split_seed,
            train_rows,
            test_rows,
            train_groups,
            test_groups,
        )

    LOGGER.info("Rows: %d (fit rows: %d)", len(df), int(fit_mask.sum()))

    block_columns: Dict[str, List[str]] = {}
    for block_name, prefix in blocks:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if not cols:
            raise ValueError(
                f"No columns found for block '{block_name}' with prefix '{prefix}'. "
                "Check --blocks or input CSV."
            )
        block_columns[block_name] = cols
        LOGGER.info("Block '%s': %d columns (prefix='%s')", block_name, len(cols), prefix)

    transformed_blocks: Dict[str, np.ndarray] = {}

    for block_name, cols in block_columns.items():
        block_data = df[cols].to_numpy(dtype=np.float32)
        fit_data = block_data[fit_mask]

        max_allowed = min(fit_data.shape[0], fit_data.shape[1])
        if max_n > max_allowed:
            raise ValueError(
                f"Requested max n_components={max_n} for block '{block_name}', "
                f"but maximum allowed is {max_allowed} (min(n_fit_rows, n_features))."
            )

        LOGGER.info("Fitting PCA for block '%s' with n_components=%d", block_name, max_n)
        pca = PCA(n_components=max_n, svd_solver=args.svd_solver)
        pca.fit(fit_data)

        explained = float(np.sum(pca.explained_variance_ratio_))
        LOGGER.info("Block '%s': total explained variance at n=%d is %.6f", block_name, max_n, explained)

        transformed = pca.transform(block_data).astype(np.float32)
        transformed_blocks[block_name] = transformed

    if args.keep_original_blocks:
        base_df = df.copy()
    else:
        cols_to_drop = []
        for cols in block_columns.values():
            cols_to_drop.extend(cols)
        base_df = df.drop(columns=cols_to_drop)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in n_values:
        out_df = base_df.copy()

        for block_name in block_columns:
            values = transformed_blocks[block_name][:, :n]
            pca_cols = [f"{block_name}_pca_{i:03d}" for i in range(n)]
            out_df = pd.concat([out_df, pd.DataFrame(values, columns=pca_cols)], axis=1)

        filename = args.output_template.format(stem=input_path.stem, n=n)
        out_path = output_dir / filename
        out_df = relativize_path_columns(out_df)
        out_df.to_csv(out_path, index=False)
        LOGGER.info("Saved PCA CSV (n=%d): %s", n, out_path)


if __name__ == "__main__":
    main()
