#!/usr/bin/env bash
set -euo pipefail

# Reproduce the full QualiSR-Lab pipeline from a clean checkout.
#
# Common overrides:
#   DATASET_SOURCE=hf|gdrive|archive|skip
#   DATASET_ARCHIVE=/path/to/grounding_dataset.zip
#   DEVICE=cpu|cuda|auto
#   PROFILE=1 PROFILE_FLOPS=1
#   FEATURE_GROUPS="fr nr vgg resnet siglip"
#   CONFIG=configs/default.json PLOTS_DIR=plots FEATURES_DIR=features

PYTHON=${PYTHON:-python}
DATASET_SOURCE=${DATASET_SOURCE:-hf}
DATASET_REPO=${DATASET_REPO:-onryabinin/QualiSR-Set120}
GDRIVE_URL=${GDRIVE_URL:-https://drive.google.com/file/d/1NeGiwWQECTZMxVhJ5ZALxQ5nzRYkz4-E/view?usp=sharing}
DATASET_ARCHIVE=${DATASET_ARCHIVE:-}

DATASET_DIR=${DATASET_DIR:-dataset}
FEATURES_DIR=${FEATURES_DIR:-features}
PLOTS_DIR=${PLOTS_DIR:-plots}
CONFIG=${CONFIG:-configs/default.json}
DEVICE=${DEVICE:-auto}

SR_METHODS=${SR_METHODS:-PASD SUPIR RealESRGAN}
REF_METHODS=${REF_METHODS:-bicubic rlfn span}
FEATURE_GROUPS=${FEATURE_GROUPS:-fr nr vgg resnet siglip}
PCA_COMPONENTS=${PCA_COMPONENTS:-5 10 25 50 75}

INSTALL_DEPS=${INSTALL_DEPS:-0}
RUN_REFERENCE=${RUN_REFERENCE:-0}
PROFILE=${PROFILE:-0}
PROFILE_FLOPS=${PROFILE_FLOPS:-0}
SAVE_SVG=${SAVE_SVG:-0}

QUALISR=("$PYTHON" -m qualisr_lab.cli)
DATASET_BASE="$DATASET_DIR"

run() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

dataset_ready_at() {
  local root=$1
  [[ -d "$root/hr" && -d "$root/lr" && -d "$root/sr" ]]
}

resolve_dataset_base() {
  if dataset_ready_at "$DATASET_DIR"; then
    DATASET_BASE="$DATASET_DIR"
    return
  fi

  local child
  for child in "$DATASET_DIR"/*; do
    if [[ -d "$child" ]] && dataset_ready_at "$child"; then
      DATASET_BASE="$child"
      return
    fi
  done

  echo "Could not find dataset layout under '$DATASET_DIR'." >&2
  echo "Expected hr/, lr/, and sr/ directories." >&2
  exit 1
}

unpack_archive() {
  local archive=$1
  mkdir -p "$DATASET_DIR"
  case "$archive" in
    *.zip)
      need_cmd unzip
      run unzip -q -o "$archive" -d "$DATASET_DIR"
      ;;
    *.tar.gz|*.tgz)
      run tar -xzf "$archive" -C "$DATASET_DIR"
      ;;
    *.tar)
      run tar -xf "$archive" -C "$DATASET_DIR"
      ;;
    *)
      echo "Unsupported dataset archive format: $archive" >&2
      exit 1
      ;;
  esac
}

download_dataset() {
  if dataset_ready_at "$DATASET_DIR"; then
    echo "Dataset already present at '$DATASET_DIR'."
    return
  fi

  if [[ -n "$DATASET_ARCHIVE" ]]; then
    unpack_archive "$DATASET_ARCHIVE"
    return
  fi

  case "$DATASET_SOURCE" in
    hf)
      need_cmd hf
      run hf download "$DATASET_REPO" --repo-type dataset --local-dir "$DATASET_DIR"
      ;;
    gdrive)
      need_cmd gdown
      need_cmd unzip
      mkdir -p "$DATASET_DIR"
      run gdown --fuzzy "$GDRIVE_URL" -O "$DATASET_DIR/grounding_dataset.zip"
      run unzip -q -o "$DATASET_DIR/grounding_dataset.zip" -d "$DATASET_DIR"
      ;;
    archive)
      if [[ -z "$DATASET_ARCHIVE" ]]; then
        echo "DATASET_SOURCE=archive requires DATASET_ARCHIVE=/path/to/archive." >&2
        exit 1
      fi
      ;;
    skip)
      echo "Skipping dataset download."
      ;;
    *)
      echo "Unknown DATASET_SOURCE='$DATASET_SOURCE'. Use hf, gdrive, archive, or skip." >&2
      exit 1
      ;;
  esac
}

sync_labels() {
  mkdir -p scores
  if [[ -f "$DATASET_BASE/labels.csv" ]]; then
    run cp "$DATASET_BASE/labels.csv" scores/labels.csv
  elif [[ -f scores/labels.csv ]]; then
    echo "Using existing scores/labels.csv."
  else
    echo "Missing labels CSV. Expected '$DATASET_BASE/labels.csv' or scores/labels.csv." >&2
    exit 1
  fi
}

build_dataset_args() {
  read -r -a METHOD_ARRAY <<< "$SR_METHODS"
  read -r -a REF_ARRAY <<< "$REF_METHODS"

  SR_ARGS=()
  HEATMAP_ARGS=()
  for method in "${METHOD_ARRAY[@]}"; do
    SR_ARGS+=("${method}=${DATASET_BASE}/sr/${method}")
    HEATMAP_ARGS+=("${method}=${DATASET_BASE}/heatmaps/${method}")
  done

  REF_ARGS=()
  for ref in "${REF_ARRAY[@]}"; do
    REF_ARGS+=("${ref}=${DATASET_BASE}/ref/${ref}")
  done
}

maybe_install_deps() {
  if [[ "$INSTALL_DEPS" == "1" ]]; then
    run "$PYTHON" -m pip install -e ".[features,regressors]"
  fi
}

maybe_make_references() {
  if [[ "$RUN_REFERENCE" != "1" ]]; then
    echo "Skipping reference generation; downloaded dataset already includes dataset/ref."
    return
  fi

  run "${QUALISR[@]}" make-reference \
    --lr-dir "$DATASET_BASE/lr" \
    --sr-dirs "${SR_ARGS[@]}" \
    --out-root "$DATASET_BASE/ref" \
    --refs bicubic rlfn span \
    --scale 4 \
    --rlfn-script "${RLFN_SCRIPT:-realtime_sr/RLFN/inference-RLFN.py}" \
    --rlfn-ckpt "${RLFN_CKPT:-realtime_sr/RLFN/rlfn-tuned-4x.pth}" \
    --span-script "${SPAN_SCRIPT:-realtime_sr/SPAN/inference-SPAN.py}" \
    --span-ckpt "${SPAN_CKPT:-realtime_sr/SPAN/span-tuned-4x.pth}"
}

extract_feature_group() {
  local feature_group=$1
  local output_csv="$FEATURES_DIR/${feature_group}.csv"
  local -a FEATURE_PROFILE_ARGS=()
  if [[ "$PROFILE" == "1" || "$PROFILE_FLOPS" == "1" ]]; then
    FEATURE_PROFILE_ARGS+=(--profile --profile-output "$FEATURES_DIR/${feature_group}_profile.csv")
  fi
  if [[ "$PROFILE_FLOPS" == "1" ]]; then
    FEATURE_PROFILE_ARGS+=(--profile-flops)
  fi

  run "${QUALISR[@]}" extract-features \
    --sr-dirs "${SR_ARGS[@]}" \
    --gt-dir "$DATASET_BASE/hr" \
    --lr-dir "$DATASET_BASE/lr" \
    --ref-dirs "${REF_ARGS[@]}" \
    --features "$feature_group" \
    --output "$output_csv" \
    --device "$DEVICE" \
    "${FEATURE_PROFILE_ARGS[@]}"
}

extract_features() {
  mkdir -p "$FEATURES_DIR"
  local -a FEATURE_GROUP_ARRAY=()
  read -r -a FEATURE_GROUP_ARRAY <<< "${FEATURE_GROUPS//,/ }"

  for feature_group in "${FEATURE_GROUP_ARRAY[@]}"; do
    extract_feature_group "$feature_group"
  done
}

apply_pca() {
  mkdir -p "$FEATURES_DIR/pca"
  read -r -a PCA_ARRAY <<< "$PCA_COMPONENTS"

  run "${QUALISR[@]}" apply-pca \
    --input "$FEATURES_DIR/vgg.csv" \
    --blocks vgg=vgg_ \
    --n-components "${PCA_ARRAY[@]}" \
    --test-size 0.2 \
    --split-seed 42 \
    --output-dir "$FEATURES_DIR/pca"

  run "${QUALISR[@]}" apply-pca \
    --input "$FEATURES_DIR/resnet.csv" \
    --blocks resnet=resnet_ \
    --n-components "${PCA_ARRAY[@]}" \
    --test-size 0.2 \
    --split-seed 42 \
    --output-dir "$FEATURES_DIR/pca"
}

compute_stats() {
  local -a STATS_PROFILE_ARGS=()
  if [[ "$PROFILE" == "1" || "$PROFILE_FLOPS" == "1" ]]; then
    STATS_PROFILE_ARGS+=(--profile --profile-output "$FEATURES_DIR/stats_profile.csv")
  fi

  run "${QUALISR[@]}" compute-stats \
    --heatmap-dirs "${HEATMAP_ARGS[@]}" \
    --output "$FEATURES_DIR/stats.csv" \
    --percentiles 5 95 \
    --area-thresholds 0 0.5 0.75 \
    "${STATS_PROFILE_ARGS[@]}"
}

run_regressors() {
  mkdir -p "$PLOTS_DIR"

  local -a REGRESSOR_ARGS=(--config "$CONFIG" --plots-root "$PLOTS_DIR")
  if [[ "$SAVE_SVG" == "1" ]]; then
    REGRESSOR_ARGS+=(--save-svg)
  fi
  if [[ "$PROFILE" == "1" || "$PROFILE_FLOPS" == "1" ]]; then
    REGRESSOR_ARGS+=(--profile)
    local -a PROFILE_FILES=()
    for profile_file in "$FEATURES_DIR"/*_profile.csv; do
      [[ -f "$profile_file" ]] && PROFILE_FILES+=("$profile_file")
    done
    if [[ ${#PROFILE_FILES[@]} -gt 0 ]]; then
      REGRESSOR_ARGS+=(--feature-profile-files "${PROFILE_FILES[@]}")
    fi
  fi

  run "${QUALISR[@]}" run-regressors "${REGRESSOR_ARGS[@]}"
}

main() {
  maybe_install_deps
  download_dataset
  resolve_dataset_base
  sync_labels
  build_dataset_args
  maybe_make_references
  extract_features
  apply_pca
  compute_stats
  run_regressors

  echo
  echo "Reproducibility pipeline complete."
  echo "Dataset: $DATASET_BASE"
  echo "Features: $FEATURES_DIR"
  echo "Plots: $PLOTS_DIR"
}

main "$@"
