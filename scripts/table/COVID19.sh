#!/usr/bin/env bash
set -euo pipefail



cuda_id="${CUDA_ID:-0}"
dataset="${DATASET:-COVID19}"
dim="${DIM:-2}"
model="${MODEL:-SMASH}"
normalize_location="${NORMALIZE_LOCATION:-1}"
normalize_time="${NORMALIZE_TIME:-0}"
batch_size="${BATCH_SIZE:-90}"

total_epochs="${TOTAL_EPOCHS:-200}"
cond_dim="${COND_DIM:-32}"
num_units="${NUM_UNITS:-64}"
alpha_CE="${ALPHA_CE:-1.}"
alpha_s="${ALPHA_S:-.5}"
grid_s="${GRID_S:-5}"
grid_t="${GRID_T:-5}"
num_noise="${NUM_NOISE:-32}"
sigma_t="${SIGMA_T:-0.5}"
sigma_s="${SIGMA_S:-0.01}"
noise_type="${NOISE_TYPE:-lognormal}"
with_survival="${WITH_SURVIVAL:-1}"

estimator="${ESTIMATOR:-wsm}"
seed="${SEED:-0}"

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
