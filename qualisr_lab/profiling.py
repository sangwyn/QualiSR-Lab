"""Runtime and FLOPs profiling helpers for QualiSR-Lab pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def is_regressor_profiling_enabled(cfg: dict[str, Any]) -> bool:
    profile_cfg = cfg.get("profiling", {})
    return bool(profile_cfg.get("regressors", False))


def resolve_profile_template_path(path_template: str, cfg: dict[str, Any], run_name: str | None = None) -> Path:
    return Path(
        path_template.format(
            features_root=cfg["paths"]["features_root"],
            plots_root=cfg["paths"]["plots_root"],
            experiment_name=cfg["experiment_name"],
            run_name=run_name or cfg["experiment_name"],
            pca_n=cfg["features"].get("pca_n", 0),
        )
    )


def resolve_regressor_profile_path(cfg: dict[str, Any], out_dir: Path, run_name: str) -> Path:
    profile_cfg = cfg.get("profiling", {})
    raw_path = profile_cfg.get("regressor_output")
    if not raw_path:
        return out_dir / "regressor_profile.csv"

    return resolve_profile_template_path(str(raw_path), cfg, run_name=run_name)


def resolve_regressor_total_profile_path(cfg: dict[str, Any], out_dir: Path, run_name: str) -> Path:
    profile_cfg = cfg.get("profiling", {})
    raw_path = profile_cfg.get("regressor_total_output")
    if not raw_path:
        return out_dir / "regressor_total_profile.csv"

    return resolve_profile_template_path(str(raw_path), cfg, run_name=run_name)


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _tree_count(model: Any) -> int:
    if hasattr(model, "estimators_"):
        return len(model.estimators_)
    if hasattr(model, "get_booster"):
        try:
            return int(model.get_booster().num_boosted_rounds())
        except (AttributeError, ValueError):
            return 0
    if hasattr(model, "tree_count_"):
        return int(model.tree_count_)
    if hasattr(model, "get_tree_leaf_counts"):
        return int(len(model.get_tree_leaf_counts()))
    return 0


def _sklearn_tree_node_counts(model: Any) -> tuple[int, int]:
    estimators = getattr(model, "estimators_", None)
    if estimators is None and hasattr(model, "tree_"):
        estimators = [model]
    if estimators is None:
        return 0, 0

    total_nodes = 0
    internal_nodes = 0
    for estimator in np.ravel(estimators):
        tree = getattr(estimator, "tree_", None)
        if tree is None:
            continue
        total_nodes += int(tree.node_count)
        internal_nodes += int(np.count_nonzero(tree.children_left != tree.children_right))
    return total_nodes, internal_nodes


def _estimate_sklearn_tree_predict_ops(model: Any, X: pd.DataFrame) -> dict[str, Any] | None:
    if not hasattr(model, "decision_path"):
        return None

    try:
        decision_path = model.decision_path(X)
    except (AttributeError, ValueError, TypeError):
        return None

    indicator = decision_path[0] if isinstance(decision_path, tuple) else decision_path
    n_samples = len(X)
    n_trees = _tree_count(model) or 1
    node_visits = int(indicator.nnz)
    comparisons = max(node_visits - n_samples * n_trees, 0)
    aggregation_ops = n_samples * n_trees
    total_ops = comparisons + aggregation_ops
    total_nodes, internal_nodes = _sklearn_tree_node_counts(model)

    return {
        "estimated_predict_flops": float(total_ops),
        "estimated_predict_comparisons": float(comparisons),
        "flops_method": "sklearn decision_path tree comparisons + ensemble aggregation",
        "n_trees": n_trees,
        "n_nodes": total_nodes,
        "n_internal_nodes": internal_nodes,
    }


def _xgb_node_id(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if not text or text == "nan":
        return None
    try:
        return int(text.rsplit("-", 1)[-1])
    except ValueError:
        return None


def _xgb_leaf_depths(tree_df: pd.DataFrame) -> dict[int, int]:
    children: dict[int, list[int]] = {}
    leaves: set[int] = set()
    for row in tree_df.itertuples(index=False):
        node = int(row.Node)
        feature = str(row.Feature)
        if feature == "Leaf":
            leaves.add(node)
            continue

        child_ids = []
        for attr in ("Yes", "No", "Missing"):
            child = _xgb_node_id(getattr(row, attr, None))
            if child is not None:
                child_ids.append(child)
        children[node] = child_ids

    depths: dict[int, int] = {}
    stack = [(0, 0)]
    seen: set[int] = set()
    while stack:
        node, depth = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in leaves:
            depths[node] = depth
        for child in children.get(node, []):
            stack.append((child, depth + 1))

    return depths


def _estimate_xgboost_predict_ops(model: Any, X: pd.DataFrame) -> dict[str, Any] | None:
    if not hasattr(model, "get_booster"):
        return None

    try:
        booster = model.get_booster()
        tree_df = booster.trees_to_dataframe()
    except (AttributeError, ValueError, TypeError):
        return None

    n_samples = len(X)
    n_trees = int(tree_df["Tree"].nunique()) if "Tree" in tree_df.columns else _tree_count(model)
    total_nodes = int(len(tree_df))
    internal_nodes = int((tree_df["Feature"] != "Leaf").sum()) if "Feature" in tree_df.columns else 0
    fallback_depths = []
    depth_by_tree: dict[int, dict[int, int]] = {}
    for tree_id, group in tree_df.groupby("Tree"):
        leaf_depths = _xgb_leaf_depths(group)
        if leaf_depths:
            depth_by_tree[int(tree_id)] = leaf_depths
            fallback_depths.append(float(np.mean(list(leaf_depths.values()))))

    fallback_depth = float(np.mean(fallback_depths)) if fallback_depths else np.nan
    comparisons = np.nan
    method = "xgboost mean leaf depth estimate + ensemble aggregation"
    try:
        import xgboost as xgb

        leaves = np.asarray(booster.predict(xgb.DMatrix(X), pred_leaf=True))
        if leaves.ndim == 1:
            leaves = leaves.reshape(n_samples, -1)

        comparison_count = 0.0
        for tree_index in range(leaves.shape[1]):
            leaf_depths = depth_by_tree.get(tree_index, {})
            tree_fallback = float(np.mean(list(leaf_depths.values()))) if leaf_depths else fallback_depth
            for leaf_id in leaves[:, tree_index]:
                comparison_count += _safe_float(leaf_depths.get(int(leaf_id), tree_fallback))
        comparisons = comparison_count
        method = "xgboost predicted leaf depths + ensemble aggregation"
    except (ImportError, ValueError, TypeError, AttributeError):
        if not np.isnan(fallback_depth):
            comparisons = n_samples * n_trees * fallback_depth

    aggregation_ops = n_samples * n_trees
    total_ops = comparisons + aggregation_ops if not np.isnan(comparisons) else np.nan
    return {
        "estimated_predict_flops": float(total_ops),
        "estimated_predict_comparisons": float(comparisons),
        "flops_method": method,
        "n_trees": n_trees,
        "n_nodes": total_nodes,
        "n_internal_nodes": internal_nodes,
    }


def _estimate_catboost_predict_ops(model: Any, X: pd.DataFrame) -> dict[str, Any] | None:
    if not hasattr(model, "get_tree_leaf_counts"):
        return None

    try:
        leaf_counts = np.asarray(model.get_tree_leaf_counts(), dtype=float)
    except (AttributeError, ValueError, TypeError):
        return None
    if leaf_counts.size == 0:
        return None

    depths = np.ceil(np.log2(np.maximum(leaf_counts, 1.0)))
    n_samples = len(X)
    n_trees = int(leaf_counts.size)
    comparisons = float(n_samples * depths.sum())
    aggregation_ops = n_samples * n_trees
    total_ops = comparisons + aggregation_ops
    internal_nodes = int(np.maximum(leaf_counts - 1, 0).sum())
    return {
        "estimated_predict_flops": float(total_ops),
        "estimated_predict_comparisons": float(comparisons),
        "flops_method": "catboost symmetric tree depth estimate + ensemble aggregation",
        "n_trees": n_trees,
        "n_nodes": int(leaf_counts.sum()) + internal_nodes,
        "n_internal_nodes": internal_nodes,
    }


def estimate_regressor_predict_flops(model_name: str, model: Any, X: pd.DataFrame) -> dict[str, Any]:
    estimators = (
        _estimate_sklearn_tree_predict_ops,
        _estimate_xgboost_predict_ops,
        _estimate_catboost_predict_ops,
    )
    for estimator in estimators:
        estimate = estimator(model, X)
        if estimate is not None:
            estimate["flops_note"] = (
                "Estimated inference operation count; tree comparisons are counted as one operation each."
            )
            return estimate

    return {
        "estimated_predict_flops": np.nan,
        "estimated_predict_comparisons": np.nan,
        "flops_method": f"not_available_for_{model_name}",
        "flops_note": "Estimator does not expose enough structure for a prediction FLOPs estimate.",
        "n_trees": _tree_count(model),
        "n_nodes": np.nan,
        "n_internal_nodes": np.nan,
    }


def build_regressor_profile_row(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_runtime_sec: float,
    predict_runtime_sec: float,
) -> dict[str, Any]:
    estimate = estimate_regressor_predict_flops(model_name, model, X_test)
    n_test = len(X_test)
    predict_flops = _safe_float(estimate["estimated_predict_flops"])
    predict_comparisons = _safe_float(estimate["estimated_predict_comparisons"])

    return {
        "model": model_name,
        "n_train_samples": len(X_train),
        "n_test_samples": n_test,
        "n_features": X_train.shape[1],
        "n_trees": estimate["n_trees"],
        "n_nodes": estimate["n_nodes"],
        "n_internal_nodes": estimate["n_internal_nodes"],
        "train_runtime_sec": train_runtime_sec,
        "predict_runtime_sec": predict_runtime_sec,
        "mean_predict_runtime_sec": predict_runtime_sec / n_test if n_test else np.nan,
        "estimated_train_flops": np.nan,
        "train_flops_method": "not_available_for_tree_ensemble_training",
        "estimated_predict_flops": predict_flops,
        "mean_predict_flops": predict_flops / n_test if n_test and not np.isnan(predict_flops) else np.nan,
        "estimated_predict_comparisons": predict_comparisons,
        "mean_predict_comparisons": (
            predict_comparisons / n_test if n_test and not np.isnan(predict_comparisons) else np.nan
        ),
        "predict_flops_method": estimate["flops_method"],
        "flops_note": estimate["flops_note"],
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _resolve_feature_path(feat_name: str, cfg: dict[str, Any]) -> Path:
    templates = cfg["features"]["feature_files"]
    if feat_name not in templates:
        raise KeyError(f"No path template configured for feature '{feat_name}'")

    return Path(
        templates[feat_name].format(
            features_root=cfg["paths"]["features_root"],
            pca_n=cfg["features"].get("pca_n", 0),
        )
    )


def infer_feature_profile_paths(feature_name: str, cfg: dict[str, Any]) -> list[Path]:
    profile_cfg = cfg.get("profiling", {})
    paths: list[Path] = []

    for item in _as_list(profile_cfg.get("feature_profile_files")):
        paths.append(resolve_profile_template_path(str(item), cfg))

    configured_profiles = profile_cfg.get("feature_profiles", {})
    for item in _as_list(configured_profiles.get(feature_name)):
        paths.append(resolve_profile_template_path(str(item), cfg))

    try:
        feature_path = _resolve_feature_path(feature_name, cfg)
    except KeyError:
        feature_path = None

    if feature_path is not None:
        paths.append(feature_path.with_name(f"{feature_path.stem}_profile.csv"))
        if "_pca" in feature_path.stem:
            base_name = feature_path.stem.split("_pca", 1)[0]
            paths.append(feature_path.parent / f"{base_name}_profile.csv")
            paths.append(feature_path.parent.parent / f"{base_name}_profile.csv")

    seen: set[Path] = set()
    existing = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def feature_profile_matches_input(feature_name: str, input_columns: set[str]) -> bool:
    if feature_name in input_columns:
        return True

    aliases = {
        "vgg": ("vgg_", "vgg_pca_"),
        "resnet": ("resnet_", "resnet_pca_"),
        "siglip": ("content_fidelity", "perceptual_enhancement", "final_rr_score"),
        "gaussian": ("gaussian_",),
        "uniform": ("uniform_",),
    }
    prefixes_or_names = aliases.get(feature_name, (f"{feature_name}_",))
    for value in prefixes_or_names:
        if value in input_columns or any(column.startswith(value) for column in input_columns):
            return True
    return False


def count_profiled_input_features(feature_name: str, input_columns: set[str]) -> int:
    if feature_name in input_columns:
        return 1

    aliases = {
        "vgg": ("vgg_", "vgg_pca_"),
        "resnet": ("resnet_", "resnet_pca_"),
        "siglip": ("content_fidelity", "perceptual_enhancement", "final_rr_score"),
        "gaussian": ("gaussian_",),
        "uniform": ("uniform_",),
    }
    prefixes_or_names = aliases.get(feature_name, (f"{feature_name}_",))
    matched: set[str] = set()
    for value in prefixes_or_names:
        if value in input_columns:
            matched.add(value)
        matched.update(column for column in input_columns if column.startswith(value))
    return len(matched)


def load_feature_profile_summary(cfg: dict[str, Any], input_columns: pd.Index) -> pd.DataFrame:
    feature_names = list(cfg["features"].get("include", []))
    if cfg["features"].get("include_stats", False):
        feature_names.append("stats")

    input_column_set = set(map(str, input_columns))
    rows = []
    seen_profile_rows: set[tuple[Path, str]] = set()
    for feature_name in feature_names:
        for profile_path in infer_feature_profile_paths(feature_name, cfg):
            profile = pd.read_csv(profile_path)
            if "feature" not in profile.columns:
                continue

            for row in profile.to_dict("records"):
                profile_feature = str(row["feature"])
                key = (profile_path, profile_feature)
                if key in seen_profile_rows or not feature_profile_matches_input(profile_feature, input_column_set):
                    continue
                seen_profile_rows.add(key)

                rows.append(
                    {
                        "profile_path": str(profile_path),
                        "profile_feature": profile_feature,
                        "matched_input_features": count_profiled_input_features(profile_feature, input_column_set),
                        "extractor_feature_count": _safe_float(row.get("feature_count")),
                        "profile_samples": _safe_float(row.get("samples")),
                        "mean_runtime_sec": _safe_float(row.get("mean_runtime_sec")),
                        "total_runtime_sec": _safe_float(row.get("total_runtime_sec")),
                        "mean_flops": _safe_float(row.get("mean_flops")),
                        "total_profiled_flops": _safe_float(row.get("total_profiled_flops")),
                        "flops_profiled_samples": _safe_float(row.get("flops_profiled_samples")),
                    }
                )

    return pd.DataFrame(rows)


def finite_sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else np.nan


def build_regressor_total_profile(
    regressor_profile: pd.DataFrame,
    feature_profile_summary: pd.DataFrame,
) -> pd.DataFrame:
    if regressor_profile.empty or feature_profile_summary.empty:
        return pd.DataFrame()

    feature_mean_runtime = finite_sum(feature_profile_summary["mean_runtime_sec"])
    feature_mean_flops = finite_sum(feature_profile_summary["mean_flops"])
    feature_profile_rows = len(feature_profile_summary)
    profiled_input_features = int(feature_profile_summary["matched_input_features"].sum())
    extractor_feature_count = finite_sum(feature_profile_summary["extractor_feature_count"])
    source_paths = ";".join(sorted(feature_profile_summary["profile_path"].unique().tolist()))
    source_features = ";".join(feature_profile_summary["profile_feature"].tolist())

    rows = []
    for row in regressor_profile.to_dict("records"):
        n_test = int(row["n_test_samples"])
        feature_total_runtime = feature_mean_runtime * n_test if not np.isnan(feature_mean_runtime) else np.nan
        feature_total_flops = feature_mean_flops * n_test if not np.isnan(feature_mean_flops) else np.nan
        model_predict_runtime = _safe_float(row.get("predict_runtime_sec"))
        model_predict_flops = _safe_float(row.get("estimated_predict_flops"))
        model_mean_runtime = _safe_float(row.get("mean_predict_runtime_sec"))
        model_mean_flops = _safe_float(row.get("mean_predict_flops"))

        rows.append(
            {
                "model": row["model"],
                "n_input_features": int(row["n_features"]),
                "n_profiled_input_features": profiled_input_features,
                "n_extractor_features": extractor_feature_count,
                "n_feature_profile_rows": feature_profile_rows,
                "n_test_samples": n_test,
                "feature_profile_sources": source_paths,
                "feature_profile_features": source_features,
                "feature_mean_runtime_sec": feature_mean_runtime,
                "feature_total_runtime_sec": feature_total_runtime,
                "feature_mean_flops": feature_mean_flops,
                "feature_total_flops": feature_total_flops,
                "regressor_train_runtime_sec": _safe_float(row.get("train_runtime_sec")),
                "regressor_predict_runtime_sec": model_predict_runtime,
                "regressor_mean_predict_runtime_sec": model_mean_runtime,
                "regressor_predict_flops": model_predict_flops,
                "regressor_mean_predict_flops": model_mean_flops,
                "pipeline_predict_total_runtime_sec": feature_total_runtime + model_predict_runtime,
                "pipeline_predict_mean_runtime_sec": feature_mean_runtime + model_mean_runtime,
                "pipeline_predict_total_flops": feature_total_flops + model_predict_flops,
                "pipeline_predict_mean_flops": feature_mean_flops + model_mean_flops,
            }
        )

    return pd.DataFrame(rows)
