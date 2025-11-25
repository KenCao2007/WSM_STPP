#!/usr/bin/env bash
set -euo pipefail

# find current .sh directory (assume in repo_root/scripts/bash/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# run_copy.py is in the upper upper directory: repo_root/
cd "${SCRIPT_DIR}/../.."
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGDIR"

# parse arguments (support --arg=value / --arg value; --seeds comma separated)
seed=""; seeds=""
dataset=""; estimator="unknown"
other_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed=*)      seed="${1#*=}"; shift ;;
    --seed)        seed="$2"; shift 2 ;;
    --seeds=*)     seeds="${1#*=}"; shift ;;
    --seeds)       seeds="$2"; shift 2 ;;
    --dataset=*)   dataset="${1#*=}"; shift ;;
    --dataset)     dataset="$2"; shift 2 ;;
    --estimator=*) estimator="${1#*=}"; shift ;;
    --estimator)   estimator="$2"; shift 2 ;;
    *)             other_args+=("$1"); shift ;;
  esac
done

# remove possible quotes (e.g. --dataset "Earthquake")
dataset="${dataset%\"}"; dataset="${dataset#\"}"
dataset="${dataset%\'}"; dataset="${dataset#\'}"

# dataset is required
if [[ -z "$dataset" ]]; then
  echo "[run.sh] ERROR: --dataset is required." >&2
  exit 1
fi

# select script based on dataset: {dataset}.sh (case-insensitive)
base="${dataset//[^A-Za-z0-9._-]/_}"
candidates=(
  "${SCRIPT_DIR}/${base}.sh"
  "${SCRIPT_DIR}/${base,,}.sh"
  "${SCRIPT_DIR}/${base^}.sh"
)
script_file=""
for f in "${candidates[@]}"; do
  if [[ -f "$f" ]]; then script_file="$f"; break; fi
done
if [[ -z "$script_file" ]]; then
  echo "[run.sh] ERROR: No script for dataset '${dataset}'. Tried:" >&2
  printf '  - %s\n' "${candidates[@]}" >&2
  exit 1
fi

# build seed list
if [[ -n "$seeds" ]]; then
  IFS=',' read -r -a seed_list <<< "$seeds"
elif [[ -n "$seed" ]]; then
  seed_list=("$seed")
elif [[ -n "${SEED:-}" ]]; then
  seed_list=("$SEED")
else
  seed_list=("NA")
fi

has_stdbuf=0
if command -v stdbuf >/dev/null 2>&1; then has_stdbuf=1; fi

for s in "${seed_list[@]}"; do
  ts=$(date +%F_%H-%M-%S)
  log="${LOGDIR}/${dataset}_${estimator}_seed${s}_${ts}.log"
  echo "[run.sh] dataset=${dataset} script=$(basename "$script_file") seed=${s} estimator=${estimator}"
  echo "[run.sh] log => ${log}"

    if [[ $has_stdbuf -eq 1 ]]; then
    ESTIMATOR="$estimator" DATASET="$dataset" SEED="$s" \
        stdbuf -oL -eL bash "$script_file" \
        --seed "$s" --dataset "$dataset" --estimator "$estimator" \
        "${other_args[@]}" |& tee "$log"
    else
    ESTIMATOR="$estimator" DATASET="$dataset" SEED="$s" \
        bash "$script_file" \
        --seed "$s" --dataset "$dataset" --estimator "$estimator" \
        "${other_args[@]}" |& tee "$log"
    fi
done