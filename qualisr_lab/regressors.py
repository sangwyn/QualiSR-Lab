"""Config-driven regressor experiments for QualiSR-Lab."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from qualisr_lab.profiling import (
    build_regressor_profile_row,
    build_regressor_total_profile,
    is_regressor_profiling_enabled,
    load_feature_profile_summary,
    resolve_regressor_profile_path,
    resolve_regressor_total_profile_path,
)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def safe_corr(y_true: Any, y_pred: Any) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true_arr.size < 2 or y_pred_arr.size < 2:
        return np.nan, np.nan
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            "safe_corr received arrays with different lengths: "
            f"y_true={y_true_arr.size}, y_pred={y_pred_arr.size}"
        )

    if np.allclose(y_true_arr, y_true_arr[0]) or np.allclose(y_pred_arr, y_pred_arr[0]):
        return np.nan, np.nan

    return (
        float(pearsonr(y_pred_arr, y_true_arr).statistic),
        float(spearmanr(y_pred_arr, y_true_arr).statistic),
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_scores_file(cfg: dict[str, Any]) -> pd.DataFrame:
    prep_cfg = cfg["score_preparation"]
    raw_scores = pd.read_csv(cfg["paths"]["raw_scores"])

    method_col = prep_cfg["method_column"]
    case_col = prep_cfg["case_column"]
    score_col = prep_cfg["score_column"]

    method_map = {str(k).lower(): v for k, v in prep_cfg["method_map"].items()}
    mapped_methods = raw_scores[method_col].astype(str).str.lower().map(method_map)

    if mapped_methods.isna().any():
        missing = sorted(raw_scores.loc[mapped_methods.isna(), method_col].astype(str).unique().tolist())
        raise ValueError(f"Missing method_map entries for: {missing}")

    suffix = prep_cfg["name_suffix"]
    names = mapped_methods.astype(str) + "/" + raw_scores[case_col].astype(str) + suffix
    prepared = pd.DataFrame(
        {
            "name": names,
            cfg["dataset"]["score_column"]: raw_scores[score_col],
        }
    )

    score_path = Path(cfg["paths"]["scores"])
    ensure_dir(score_path.parent)
    prepared.to_csv(score_path, index=False)
    return prepared


def load_scores(cfg: dict[str, Any]) -> pd.DataFrame:
    if cfg["score_preparation"]["enabled"]:
        scores = prepare_scores_file(cfg)
    else:
        scores = pd.read_csv(cfg["paths"]["scores"])

    name_col = cfg["dataset"]["name_column"]
    score_col = cfg["dataset"]["score_column"]

    if name_col not in scores.columns:
        raise ValueError(f"Scores file must contain '{name_col}' column")

    if score_col not in scores.columns:
        fallback = [c for c in ["score", "scores", "mos", "mos_norm"] if c in scores.columns]
        if not fallback:
            raise ValueError(
                f"Scores file must contain '{score_col}' column. Available: {scores.columns.tolist()}"
            )
        scores = scores.rename(columns={fallback[0]: score_col})

    return scores[[name_col, score_col]].copy()


def build_sample_name(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    method_col = cfg["dataset"]["sr_method_column"]
    filename_col = cfg["dataset"]["sr_filename_column"]
    suffix = cfg["dataset"]["filename_suffix"]

    if method_col not in df.columns or filename_col not in df.columns:
        raise ValueError(
            f"Feature file must have '{method_col}' and '{filename_col}' columns. "
            f"Got: {df.columns.tolist()}"
        )

    stem = df[filename_col].astype(str).str.rsplit(".", n=1).str[0]
    return df[method_col].astype(str) + "/" + stem + suffix


def resolve_feature_path(feat_name: str, cfg: dict[str, Any]) -> Path:
    templates = cfg["features"]["feature_files"]
    if feat_name not in templates:
        raise KeyError(f"No path template configured for feature '{feat_name}'")

    try:
        pca_n = cfg["features"]["pca_n"]
    except KeyError:
        pca_n = 0

    return Path(
        templates[feat_name].format(
            features_root=cfg["paths"]["features_root"],
            pca_n=pca_n,
        )
    )


def resolve_configured_path(path_template: str, cfg: dict[str, Any]) -> Path:
    return Path(
        path_template.format(
            features_root=cfg["paths"]["features_root"],
            pca_n=cfg["features"]["pca_n"],
        )
    )


def keep_requested_fr_columns(df: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    refs_lower = [r.lower() for r in refs]
    keep_cols = ["name"]
    for col in df.columns:
        if col == "name":
            continue
        if any(col.lower().endswith("_" + ref) for ref in refs_lower):
            keep_cols.append(col)
    return df[keep_cols]


def load_feature_block(feat_name: str, cfg: dict[str, Any], valid_names: set[str]) -> pd.DataFrame:
    path = resolve_feature_path(feat_name, cfg)
    if not path.exists():
        raise FileNotFoundError(f"Feature file for '{feat_name}' not found: {path}")

    df = pd.read_csv(path)
    df["name"] = build_sample_name(df, cfg)

    drop_candidates = cfg["dataset"]["metadata_drop"]
    drop_existing = [c for c in drop_candidates if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    if feat_name == "fr":
        df = keep_requested_fr_columns(df, cfg["features"]["fr_refs"])

    return df[df["name"].isin(valid_names)].copy()


def load_stats_block(cfg: dict[str, Any], valid_names: set[str]) -> pd.DataFrame:
    stats_path = resolve_feature_path("stats", cfg)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    stats = pd.read_csv(stats_path)
    requested = ["name"] + cfg["features"]["stats_columns"]
    missing = [c for c in requested if c not in stats.columns]
    if missing:
        raise ValueError(f"Stats file is missing requested columns: {missing}")

    stats = stats[requested]
    return stats[stats["name"].isin(valid_names)].copy()


def build_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    scores = load_scores(cfg)
    valid_names = set(scores[cfg["dataset"]["name_column"]].tolist())

    frames = [scores]
    if cfg["features"]["include_stats"]:
        frames.append(load_stats_block(cfg, valid_names))

    for feat_name in cfg["features"]["include"]:
        frames.append(load_feature_block(feat_name, cfg, valid_names))

    dataset = reduce(lambda left, right: pd.merge(left, right, on="name", how="inner"), frames)

    try:
        existing_excludes = [c for c in cfg["features"]["exclude_columns"] if c in dataset.columns]
    except KeyError:
        existing_excludes = []
    if existing_excludes:
        dataset = dataset.drop(columns=existing_excludes)

    return dataset.sort_values("name").reset_index(drop=True)


def metric_comparison_column(item: dict[str, Any]) -> str:
    if "column" in item:
        return str(item["column"])

    metric = item.get("metric")
    if not metric:
        raise ValueError(f"Correlation metric item must define 'column' or 'metric': {item}")

    reference = item.get("reference")
    if reference:
        return f"{metric}_{reference}"
    return str(metric)


def metric_comparison_label(item: dict[str, Any], column: str) -> str:
    if "label" in item:
        return str(item["label"])

    feature = str(item.get("feature", "")).upper()
    reference = item.get("reference")
    metric = str(item.get("metric", column)).upper()
    if reference:
        metric = f"{metric}+{str(reference).upper()}"
    if feature:
        return f"{metric} ({feature})"
    return metric


def load_metric_comparison_values(
    item: dict[str, Any],
    cfg: dict[str, Any],
    target_names: pd.Series,
) -> tuple[str, str, pd.Series]:
    if "path" in item:
        path = resolve_configured_path(item["path"], cfg)
    else:
        feature_name = item.get("feature")
        if feature_name is None:
            raise ValueError(f"Correlation metric item must define 'feature' or 'path': {item}")
        path = resolve_feature_path(str(feature_name), cfg)

    if not path.exists():
        raise FileNotFoundError(f"Correlation metric feature file not found: {path}")

    column = metric_comparison_column(item)
    values = pd.read_csv(path)
    values["name"] = build_sample_name(values, cfg)
    if column not in values.columns:
        raise ValueError(
            f"Correlation metric column '{column}' not found in {path}. "
            f"Available columns: {values.columns.tolist()}"
        )

    subset = values[["name", column]].copy()
    if subset["name"].duplicated().any():
        duplicates = sorted(subset.loc[subset["name"].duplicated(), "name"].unique().tolist())
        raise ValueError(f"Correlation metric file {path} has duplicate sample names: {duplicates[:10]}")

    aligned = target_names.to_frame(name="name").merge(subset, on="name", how="left")[column]
    label = metric_comparison_label(item, column)
    return label, column, aligned


def normalize_metric_values(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return numeric

    value_min = finite.min()
    value_max = finite.max()
    if np.isclose(value_min, value_max):
        normalized = pd.Series(np.nan, index=numeric.index, dtype=float)
    else:
        normalized = (numeric - value_min) / (value_max - value_min)

    if not higher_is_better:
        normalized = 1 - normalized

    return normalized


def compute_metric_comparisons(
    cfg: dict[str, Any],
    dataset: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict[str, Any]]:
    comparison_cfg = cfg.get("correlation_metrics", {})
    if not comparison_cfg.get("enabled", False):
        return []

    name_col = cfg["dataset"]["name_column"]
    target_names = dataset.loc[y_test.index, name_col].reset_index(drop=True)
    target_scores = y_test.reset_index(drop=True)

    rows = []
    for item in comparison_cfg.get("items", []):
        label, column, values = load_metric_comparison_values(item, cfg, target_names)
        higher_is_better = item.get("higher_is_better", True)
        normalized_values = normalize_metric_values(values, higher_is_better=higher_is_better)

        valid = target_scores.notna() & normalized_values.notna()
        if not valid.any():
            raise ValueError(f"Correlation metric '{label}' has no values aligned with the test split")

        plcc, srcc = safe_corr(target_scores[valid], normalized_values[valid])
        rows.append(
            {
                "model": label,
                "plcc": plcc,
                "srcc": srcc,
                "source": "metric",
                "feature": item.get("feature"),
                "column": column,
                "higher_is_better": higher_is_better,
                "normalized": True,
            }
        )

    return rows


def build_group_keys(names: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    segment_idx = cfg["dataset"]["group_segment_index"]
    remove_suffix = cfg["dataset"].get("group_remove_suffix", "")

    def one_name_to_group(value: str) -> str:
        parts = str(value).split("/")
        key = parts[segment_idx] if len(parts) > segment_idx else Path(value).name
        if remove_suffix and key.endswith(remove_suffix):
            key = key[: -len(remove_suffix)]
        return key

    return names.map(one_name_to_group)


def split_dataset(
    dataset: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = dataset[cfg["dataset"]["score_column"]]
    X = dataset.drop(columns=[cfg["dataset"]["name_column"], cfg["dataset"]["score_column"]])
    X = X.apply(pd.to_numeric, errors="raise")

    groups = build_group_keys(dataset[cfg["dataset"]["name_column"]], cfg)
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"], random_state=cfg["seed"])
    train_idx, test_idx = next(splitter.split(dataset, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    if cfg["scale_features"]:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test


def feature_family(feature_name: str) -> str:
    feature = feature_name.lower()
    if any(x in feature for x in ["musiq", "arniqa", "qalign", "unique", "paq2piq"]):
        return "NR"
    if any(feature.endswith("_" + ref) for ref in ["gt", "bicubic", "span", "rlfn"]):
        return "FR"
    if feature.startswith("vgg_"):
        return "VGG"
    if feature.startswith("resnet_"):
        return "ResNet"
    if feature in {"content_fidelity", "perceptual_enhancement", "final_rr_score"}:
        return "SigLIP"
    if feature in {"min", "max", "mean", "median", "std", "p05", "p95", "area00", "area05", "area075"}:
        return "Stats"
    return "Other"


def importance_palette() -> dict[str, str]:
    return {
        "NR": "#ff6150",
        "FR": "#f8aa4b",
        "VGG": "#54d2d2",
        "ResNet": "#0e4a95",
        "SigLIP": "#5255ea",
        "Stats": "#5bea52",
        "Other": "#000000",
    }


def importance_legend_labels() -> dict[str, str]:
    return {
        "NR": "NR metrics",
        "FR": "FR metrics",
        "VGG": "VGG features",
        "ResNet": "ResNet features",
        "SigLIP": "SigLIP features",
        "Stats": "Artifact statistics",
        "Other": "Other",
    }


def _missing_optional(package_name: str, extra_name: str) -> ImportError:
    return ImportError(
        f"Optional dependency '{package_name}' is required for this enabled model. "
        f"Install it with `pip install -e .[{extra_name}]` or disable the model in the config."
    )


def plot_rc_params(cfg: dict[str, Any]) -> dict[str, Any]:
    font_size = cfg.get("plot", {}).get("font_size")
    return {"font.size": font_size} if font_size is not None else {}


def save_plot(fig: plt.Figure, out_path: Path, cfg: dict[str, Any]) -> None:
    savefig_kwargs = {}
    dpi = cfg.get("plot", {}).get("dpi")
    if dpi is not None:
        savefig_kwargs["dpi"] = dpi
    fig.savefig(out_path, **savefig_kwargs)
    if cfg.get("plot", {}).get("save_svg", False):
        fig.savefig(out_path.with_suffix(".svg"), **savefig_kwargs)


def init_models(cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    models_cfg = cfg["models"]
    seed = cfg["seed"]
    initialized: list[tuple[str, Any]] = []

    if models_cfg.get("randomforest", {}).get("enabled", False):
        params = {"random_state": seed}
        params.update(models_cfg["randomforest"].get("params", {}))
        initialized.append(("randomforest", RandomForestRegressor(**params)))

    if models_cfg.get("xgb", {}).get("enabled", False):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise _missing_optional("xgboost", "regressors") from exc

        params = {"random_state": seed, "verbosity": 0}
        params.update(models_cfg["xgb"].get("params", {}))
        initialized.append(("xgb", xgb.XGBRegressor(**params)))

    if models_cfg.get("catboost", {}).get("enabled", False):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise _missing_optional("catboost", "regressors") from exc

        params = {"random_state": seed, "verbose": 0}
        params.update(models_cfg["catboost"].get("params", {}))
        initialized.append(("catboost", CatBoostRegressor(**params)))

    if not initialized:
        raise ValueError("No models are enabled in config['models']")

    return initialized


def plot_importance(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if not hasattr(model, "feature_importances_"):
        return None

    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=cfg["permutation_repeats"],
        random_state=cfg["seed"],
        n_jobs=2,
    )

    palette = importance_palette()
    colors = [palette[feature_family(name)] for name in importances.index]

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["importance_figsize"]))
        importances.plot.barh(yerr=perm.importances_std, ax=ax, color=colors)
        ax.set_title(f"Feature Importances: {model_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()

    out_path = out_dir / f"importance_{model_name}.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def plot_all_importances(
    importance_paths: dict[str, str | None],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    valid = []
    for model_name, path in importance_paths.items():
        if path is None:
            continue
        existing_path = Path(path)
        if existing_path.exists():
            valid.append((model_name, existing_path))

    if not valid:
        return None

    images = [plt.imread(path) for _, path in valid]
    single_w, single_h = tuple(cfg["plot"]["importance_figsize"])
    with plt.rc_context(plot_rc_params(cfg)):
        fig, axes = plt.subplots(1, len(images), figsize=(single_w * len(images), single_h))

        if len(images) == 1:
            axes = [axes]

        for ax, (_, _), image in zip(axes, valid, images, strict=False):
            ax.imshow(image)
            ax.axis("off")

        palette = importance_palette()
        labels = importance_legend_labels()
        handles = [mpatches.Patch(color=palette[key], label=labels[key]) for key in labels]

        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5))
        fig.tight_layout(rect=(0, 0, 0.9, 1))

    out_path = out_dir / "all_models_importances.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def plot_correlations(
    results_df: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
    filename: str = "correlations.png",
    title: str = "Regressor and Metric Correlation Scores",
) -> Path:
    df = results_df.sort_values("srcc", ascending=False).reset_index(drop=True)

    bar_width = 0.18
    x = np.arange(len(df))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=tuple(cfg["plot"]["correlation_figsize"]))
        ax.bar(x - bar_width / 2, df["plcc"], width=bar_width, label="PLCC", color="#845ec2")
        ax.bar(x + bar_width / 2, df["srcc"], width=bar_width, label="SRCC", color="#00c9a7")
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"].tolist(), rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.tight_layout()

    out_path = out_dir / filename
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def compute_feature_correlations(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    target = pd.to_numeric(y_test.reset_index(drop=True), errors="coerce")
    rows = []

    for feature_name in X_test.columns:
        values = pd.to_numeric(X_test[feature_name].reset_index(drop=True), errors="coerce")
        valid = target.notna() & values.notna()
        if valid.any():
            plcc, srcc = safe_corr(target[valid], values[valid])
        else:
            plcc, srcc = np.nan, np.nan

        rows.append(
            {
                "feature": feature_name,
                "family": feature_family(feature_name),
                "plcc": plcc,
                "srcc": srcc,
                "abs_plcc": abs(plcc) if not np.isnan(plcc) else np.nan,
                "abs_srcc": abs(srcc) if not np.isnan(srcc) else np.nan,
            }
        )

    mean_values = X_test.apply(pd.to_numeric, errors="coerce").mean(axis=1).reset_index(drop=True)
    valid = target.notna() & mean_values.notna()
    if valid.any():
        plcc, srcc = safe_corr(target[valid], mean_values[valid])
    else:
        plcc, srcc = np.nan, np.nan
    rows.append(
        {
            "feature": "mean_features",
            "family": "Mean",
            "plcc": plcc,
            "srcc": srcc,
            "abs_plcc": abs(plcc) if not np.isnan(plcc) else np.nan,
            "abs_srcc": abs(srcc) if not np.isnan(srcc) else np.nan,
        }
    )

    return pd.DataFrame(rows).sort_values("abs_srcc", ascending=False).reset_index(drop=True)


def plot_feature_correlations(
    feature_correlations: pd.DataFrame,
    results_df: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if feature_correlations.empty:
        return None

    feature_rows = feature_correlations.rename(columns={"feature": "name", "family": "group"}).copy()
    feature_rows["kind"] = "feature"
    feature_rows.loc[feature_rows["group"] == "Mean", "kind"] = "mean_features"

    if "source" in results_df.columns:
        regressor_results = results_df[results_df["source"] == "regressor"].copy()
    else:
        regressor_results = pd.DataFrame()

    if not regressor_results.empty:
        regressor_rows = regressor_results.rename(columns={"model": "name"}).copy()
        regressor_rows["group"] = "Regressor"
        regressor_rows["kind"] = "regressor"
        plot_df = pd.concat(
            [
                feature_rows[["name", "group", "kind", "plcc", "srcc"]],
                regressor_rows[["name", "group", "kind", "plcc", "srcc"]],
            ],
            ignore_index=True,
        )
    else:
        plot_df = feature_rows[["name", "group", "kind", "plcc", "srcc"]]

    plot_df["abs_srcc"] = plot_df["srcc"].abs()
    plot_df = plot_df.sort_values(["kind", "abs_srcc"], ascending=[True, True]).reset_index(drop=True)

    y = np.arange(len(plot_df))
    bar_height = 0.36
    is_regressor = plot_df["kind"] == "regressor"
    is_mean_features = plot_df["kind"] == "mean_features"
    plcc_colors = np.select(
        [is_regressor, is_mean_features],
        ["#ffb347", "#8ecae6"],
        default="#b8c0cc",
    )
    srcc_colors = np.select(
        [is_regressor, is_mean_features],
        ["#e85d04", "#219ebc"],
        default="#4d5a68",
    )

    default_height = max(6, 0.34 * len(plot_df))
    figsize = tuple(cfg.get("plot", {}).get("feature_correlation_figsize", [12, default_height]))

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y - bar_height / 2, plot_df["plcc"], height=bar_height, color=plcc_colors)
        ax.barh(y + bar_height / 2, plot_df["srcc"], height=bar_height, color=srcc_colors)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["name"].tolist())
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_xlabel("Correlation")
        ax.set_title("Feature, Mean Feature, and Regressor Correlations")

        handles = [
            mpatches.Patch(color="#b8c0cc", label="Feature PLCC"),
            mpatches.Patch(color="#4d5a68", label="Feature SRCC"),
            mpatches.Patch(color="#8ecae6", label="Mean Features PLCC"),
            mpatches.Patch(color="#219ebc", label="Mean Features SRCC"),
            mpatches.Patch(color="#ffb347", label="Regressor PLCC"),
            mpatches.Patch(color="#e85d04", label="Regressor SRCC"),
        ]
        ax.legend(handles=handles, loc="lower right")
        fig.tight_layout()

    out_path = out_dir / "feature_correlations.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def feature_matrix_for_cross_correlation(X: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    numeric = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_features = cfg.get("plot", {}).get("max_feature_correlation_matrix_features", 300)
    if max_features is None or numeric.shape[1] <= int(max_features):
        return numeric

    variances = numeric.var(axis=0, skipna=True).sort_values(ascending=False)
    selected = variances.head(int(max_features)).index.tolist()
    return numeric[selected]


def compute_feature_cross_correlations(X: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    matrix = feature_matrix_for_cross_correlation(X, cfg)
    return matrix.corr(method="pearson")


def plot_feature_cross_correlation_matrix(
    cross_correlations: pd.DataFrame,
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path | None:
    if cross_correlations.empty:
        return None

    n_features = len(cross_correlations)
    default_size = min(max(7.0, 0.34 * n_features), 24.0)
    figsize = tuple(cfg.get("plot", {}).get("feature_correlation_matrix_figsize", [default_size, default_size]))
    label_limit = int(cfg.get("plot", {}).get("max_feature_correlation_matrix_labels", 45))
    show_labels = n_features <= label_limit
    label_font_size = cfg.get("plot", {}).get(
        "feature_correlation_matrix_label_font_size",
        min(10, plt.rcParams["font.size"]),
    )

    values = np.ma.masked_invalid(cross_correlations.to_numpy(dtype=float))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")

    with plt.rc_context(plot_rc_params(cfg)):
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, interpolation="nearest")
        # title = "Feature Cross-Correlation Matrix"
        # if show_labels:
        #     fig.suptitle(title, y=0.98)
        # else:
        #     ax.set_title(title)

        if show_labels:
            ticks = np.arange(n_features)
            labels = cross_correlations.columns.tolist()
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=label_font_size)
            ax.set_yticklabels(labels, fontsize=label_font_size)
            ax.tick_params(
                axis="x",
                which="both",
                top=True,
                bottom=True,
                labeltop=True,
                labelbottom=True,
                pad=2,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=True,
                labelleft=True,
                labelright=True,
                pad=2,
            )
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_rotation(90)
                tick.label1.set_ha("center")
                tick.label1.set_va("top")
                tick.label1.set_fontsize(label_font_size)
                tick.label2.set_rotation(90)
                tick.label2.set_ha("center")
                tick.label2.set_va("bottom")
                tick.label2.set_fontsize(label_font_size)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_ha("right")
                tick.label2.set_ha("left")
                tick.label1.set_fontsize(label_font_size)
                tick.label2.set_fontsize(label_font_size)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{n_features} features")
            ax.set_ylabel(f"{n_features} features")

        if show_labels:
            fig.subplots_adjust(left=0.18, right=0.70, bottom=0.22, top=0.74)
            cbar_ax = fig.add_axes([0.88, 0.22, 0.025, 0.52])
            cbar = fig.colorbar(image, cax=cbar_ax)
        else:
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
        cbar.set_label("Pearson correlation")

    out_path = out_dir / "feature_cross_correlation_matrix.png"
    save_plot(fig, out_path, cfg)
    plt.close(fig)
    return out_path


def run_experiment(cfg: dict[str, Any], make_plots: bool = True) -> dict[str, Any]:
    np.random.seed(cfg["seed"])

    dataset = build_dataset(cfg)
    X_train, X_test, y_train, y_test = split_dataset(dataset, cfg)

    try:
        run_name = f"{cfg['experiment_name']}@pca{cfg['features']['pca_n']}"
    except KeyError:
        run_name = f"{cfg['experiment_name']}"
    out_dir = ensure_dir(Path(cfg["paths"]["plots_root"]) / run_name)

    if cfg.get("save_dataset_snapshot", False):
        dataset.to_csv(out_dir / "dataset_snapshot.csv", index=False)

    results = []
    importance_paths: dict[str, str | None] = {}
    all_plcc: list[float] = []
    all_srcc: list[float] = []
    profile_regressors = is_regressor_profiling_enabled(cfg)
    profile_rows: list[dict[str, Any]] = []

    for model_name, model in init_models(cfg):
        if profile_regressors:
            train_start = time.perf_counter()
            model.fit(X_train, y_train)
            train_runtime_sec = time.perf_counter() - train_start

            predict_start = time.perf_counter()
            pred = model.predict(X_test)
            predict_runtime_sec = time.perf_counter() - predict_start
            profile_rows.append(
                build_regressor_profile_row(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    train_runtime_sec=train_runtime_sec,
                    predict_runtime_sec=predict_runtime_sec,
                )
            )
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        plcc, srcc = safe_corr(y_test, pred)
        all_plcc.append(plcc)
        all_srcc.append(srcc)

        results.append({"model": model_name, "plcc": plcc, "srcc": srcc, "source": "regressor"})
        if make_plots:
            imp_path = plot_importance(model_name, model, X_test, y_test, out_dir, cfg)
            importance_paths[model_name] = str(imp_path) if imp_path else None

    results.extend(compute_metric_comparisons(cfg, dataset, y_test))

    if cfg["save_mean_correlations"]:
        results.append(
            {
                "model": "mean",
                "plcc": float(np.mean(all_plcc)),
                "srcc": float(np.mean(all_srcc)),
                "source": "summary",
            }
        )
    if cfg["save_best_correlations"]:
        results.append(
            {
                "model": "best",
                "plcc": float(np.max(all_plcc)),
                "srcc": float(np.max(all_srcc)),
                "source": "summary",
            }
        )

    results_df = pd.DataFrame(results).sort_values("srcc", ascending=False).reset_index(drop=True)
    results_df.to_csv(out_dir / "correlations.csv", index=False)
    regressor_profile_path = None
    regressor_total_profile_path = None
    feature_profile_summary_path = None
    if profile_rows:
        regressor_profile_path = resolve_regressor_profile_path(cfg, out_dir, run_name)
        ensure_dir(regressor_profile_path.parent)
        regressor_profile = pd.DataFrame(profile_rows)
        regressor_profile.to_csv(regressor_profile_path, index=False)

        feature_profile_summary = load_feature_profile_summary(cfg, X_train.columns)
        if not feature_profile_summary.empty:
            feature_profile_summary_path = out_dir / "regressor_feature_profile_summary.csv"
            feature_profile_summary.to_csv(feature_profile_summary_path, index=False)

            total_profile = build_regressor_total_profile(regressor_profile, feature_profile_summary)
            if not total_profile.empty:
                regressor_total_profile_path = resolve_regressor_total_profile_path(cfg, out_dir, run_name)
                ensure_dir(regressor_total_profile_path.parent)
                total_profile.to_csv(regressor_total_profile_path, index=False)

    feature_correlations = compute_feature_correlations(X_test, y_test)
    feature_correlations.to_csv(out_dir / "feature_correlations.csv", index=False)
    X_all = pd.concat([X_train, X_test], axis=0).sort_index()
    feature_cross_correlations = compute_feature_cross_correlations(X_all, cfg)
    feature_cross_correlations.to_csv(out_dir / "feature_cross_correlations.csv")

    combined_importance_path = None
    correlations_path = None
    correlations_without_metrics_path = None
    feature_correlations_path = None
    feature_cross_correlations_path = None
    if make_plots:
        combined_importance_path = plot_all_importances(importance_paths, out_dir, cfg)
        correlations_path = plot_correlations(results_df, out_dir, cfg)
        feature_correlations_path = plot_feature_correlations(feature_correlations, results_df, out_dir, cfg)
        feature_cross_correlations_path = plot_feature_cross_correlation_matrix(
            feature_cross_correlations,
            out_dir,
            cfg,
        )
        if "source" in results_df.columns:
            without_metrics = results_df[results_df["source"] != "metric"].copy()
        else:
            without_metrics = results_df.copy()
        if not without_metrics.empty:
            correlations_without_metrics_path = plot_correlations(
                without_metrics,
                out_dir,
                cfg,
                filename="correlations_without_metrics.png",
                title="Regressor Correlation Scores",
            )

    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    return {
        "dataset": dataset,
        "results": results_df,
        "output_dir": out_dir,
        "importance_paths": importance_paths,
        "all_importances_path": str(combined_importance_path) if combined_importance_path else None,
        "correlations_path": str(correlations_path) if correlations_path else None,
        "correlations_without_metrics_path": (
            str(correlations_without_metrics_path) if correlations_without_metrics_path else None
        ),
        "feature_correlations_path": str(feature_correlations_path) if feature_correlations_path else None,
        "feature_cross_correlations_path": (
            str(feature_cross_correlations_path) if feature_cross_correlations_path else None
        ),
        "regressor_profile_path": str(regressor_profile_path) if regressor_profile_path else None,
        "regressor_total_profile_path": (
            str(regressor_total_profile_path) if regressor_total_profile_path else None
        ),
        "feature_profile_summary_path": str(feature_profile_summary_path) if feature_profile_summary_path else None,
    }


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured QualiSR-Lab regressor experiments.")
    parser.add_argument("--config", default="configs/default.json", help="Path to experiment JSON config.")
    parser.add_argument("--experiment-name", default=None, help="Override config experiment_name.")
    parser.add_argument("--plots-root", default=None, help="Override config paths.plots_root.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--save-svg", action="store_true", help="Also save generated plots in SVG format.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Measure regressor train/predict runtime and save regressor_profile.csv.",
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        help=(
            "Output CSV path for regressor runtime/FLOPs profile. "
            "Implies --profile. Default: <run_output>/regressor_profile.csv."
        ),
    )
    parser.add_argument(
        "--profile-total-output",
        default=None,
        help=(
            "Output CSV path for feature+regressor runtime/FLOPs totals when feature profile data exists. "
            "Default: <run_output>/regressor_total_profile.csv."
        ),
    )
    parser.add_argument(
        "--feature-profile-files",
        nargs="+",
        default=None,
        help=(
            "Existing feature profile CSV files to aggregate into regressor totals. "
            "Can also be configured as profiling.feature_profile_files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))

    overrides: dict[str, Any] = {}
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.plots_root is not None:
        overrides.setdefault("paths", {})["plots_root"] = args.plots_root
    if args.save_svg:
        overrides.setdefault("plot", {})["save_svg"] = True
    if (
        args.profile
        or args.profile_output is not None
        or args.profile_total_output is not None
        or args.feature_profile_files is not None
    ):
        overrides.setdefault("profiling", {})["regressors"] = True
    if args.profile_output is not None:
        overrides.setdefault("profiling", {})["regressor_output"] = args.profile_output
    if args.profile_total_output is not None:
        overrides.setdefault("profiling", {})["regressor_total_output"] = args.profile_total_output
    if args.feature_profile_files is not None:
        overrides.setdefault("profiling", {})["feature_profile_files"] = args.feature_profile_files
    if overrides:
        cfg = deep_update(cfg, overrides)

    result = run_experiment(cfg, make_plots=not args.no_plots)
    print(f"Saved results to {result['output_dir']}")
    print(result["results"].to_string(index=False))


if __name__ == "__main__":
    main()
