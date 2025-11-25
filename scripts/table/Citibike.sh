#!/usr/bin/env bash
set -euo pipefail

# cd ../

# Default values, can be overridden by environment variables (CUDA_ID/SEED/DATASET etc.)
cuda_id="${CUDA_ID:-0}"
dataset="${DATASET:-Citibike}"
dim="${DIM:-2}"                     # if need to pass to run.py, add corresponding flag
model="${MODEL:-SMASH}"
normalize_location="${NORMALIZE_LOCATION:-1}"
normalize_time="${NORMALIZE_TIME:-0}"
batch_size="${BATCH_SIZE:-60}"

total_epochs="${TOTAL_EPOCHS:-150}"
cond_dim="${COND_DIM:-32}"
num_units="${NUM_UNITS:-64}"
alpha_CE="${ALPHA_CE:-1.}"
alpha_s="${ALPHA_S:-10.}"
grid_s="${GRID_S:-5}"
grid_t="${GRID_T:-5}"
num_noise="${NUM_NOISE:-10}"
sigma_t="${SIGMA_T:-0.5}"
sigma_s="${SIGMA_S:-0.05}"
noise_type="${NOISE_TYPE:-lognormal}"
with_survival="${WITH_SURVIVAL:-1}"

estimator="${ESTIMATOR:-wsm}"       # only as default value, if run.py doesn't receive, don't pass
seed="${SEED:-1}"

# Default parameters first, external passed "$@" put later —— latter overrides former
python -u run_copy.py \
  --cuda_id "${cuda_id}" \
  --dataset "${dataset}" \
  --mode train \
  --model "${model}" \
  --total_epochs "${total_epochs}" \
  --batch_size "${batch_size}" \
  --cond_dim "${cond_dim}" \
  --num_units "${num_units}" \
  --alpha_CE "${alpha_CE}" \
  --alpha_s "${alpha_s}" \
  --grid_t "${grid_t}" \
  --grid_s "${grid_s}" \
  --normalize_location "${normalize_location}" \
  --normalize_time "${normalize_time}" \
  --seed "${seed}" \
  --num_noise "${num_noise}" \
  --sigma_t "${sigma_t}" \
  --sigma_s "${sigma_s}" \
  --noise_type "${noise_type}" \
  --with_survival "${with_survival}" \
  "$@"