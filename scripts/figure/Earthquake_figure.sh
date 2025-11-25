
#!/usr/bin/env bash
set -euo pipefail

# find current .sh directory (assume in repo_root/scripts/bash/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# run_copy.py is in the upper upper directory: repo_root/
cd "${SCRIPT_DIR}/../.."

# ---------- Default values, can be overridden by environment variables ----------
seed="${SEED:-1}"
model="${MODEL:-SMASH_decouple}"     # SMASH / SMASH_decouple / SMASH_GMM
mode="${MODE:-train}"               # train / debug / test
total_epochs="${TOTAL_EPOCHS:-150}"

dim="${DIM:-2}"                     # 1 / 2 / 3 / 4
dataset="${DATASET:-Earthquake}"    # Earthquake / Crime / football / Gaussian / ...
batch_size="${BATCH_SIZE:-96}"

cuda_id="${CUDA_ID:-0}"
save_path="${SAVE_PATH:-./ModelSave/debug/}"

cond_dim="${COND_DIM:-48}"
estimator="${ESTIMATOR:-wsm}"       # wsm / mle / dsm
normalize_location="${NORMALIZE_LOCATION:-1}"
normalize_time="${NORMALIZE_TIME:-1}"

alpha_CE="${ALPHA_CE:-1.0}"
alpha_s="${ALPHA_S:-5.0}"
grid_t="${GRID_T:-10}"
grid_s="${GRID_S:-10}"
num_units="${NUM_UNITS:-48}"
num_types="${NUM_TYPES:-1}"

noise_type="${NOISE_TYPE:-lognormal}"
num_noise="${NUM_NOISE:-10}"
sigma_t="${SIGMA_T:-0.5}"
sigma_s="${SIGMA_S:-0.1}"
with_survival="${WITH_SURVIVAL:-0}"

K_trig="${K_TRIG:-2}"
spatial_weight_looser_factor="${SPATIAL_WEIGHT_LOOSER_FACTOR:-0.1}"
identity_weight="${IDENTITY_WEIGHT:-0}"

n_head="${N_HEAD:-4}"
n_layers="${N_LAYERS:-4}"
d_k="${D_K:-16}"
d_v="${D_V:-16}"
lr="${LR:-1e-3}"

eval_grid_t="${EVAL_GRID_T:-25}"
eval_grid_s="${EVAL_GRID_S:-10}"
intensity_grid_t="${INTENSITY_GRID_T:-25}"
intensity_grid_s="${INTENSITY_GRID_S:-20}"

# ---------- Call main script ----------
python run_copy.py \
  --seed "${seed}" \
  --model "${model}" \
  --mode "${mode}" \
  --total_epochs "${total_epochs}" \
  --dim "${dim}" \
  --dataset "${dataset}" \
  --batch_size "${batch_size}" \
  --cuda_id "${cuda_id}" \
  --save_path "${save_path}" \
  --cond_dim "${cond_dim}" \
  --estimator "${estimator}" \
  --normalize_location "${normalize_location}" \
  --alpha_CE "${alpha_CE}" \
  --alpha_s "${alpha_s}" \
  --grid_t "${grid_t}" \
  --grid_s "${grid_s}" \
  --num_units "${num_units}" \
  --num_types "${num_types}" \
  --normalize_time "${normalize_time}" \
  --noise_type "${noise_type}" \
  --num_noise "${num_noise}" \
  --sigma_t "${sigma_t}" \
  --sigma_s "${sigma_s}" \
  --with_survival "${with_survival}" \
  --K_trig "${K_trig}" \
  --spatial_weight_looser_factor "${spatial_weight_looser_factor}" \
  --identity_weight "${identity_weight}" \
  --n_head "${n_head}" \
  --n_layers "${n_layers}" \
  --d_k "${d_k}" \
  --d_v "${d_v}" \
  --lr "${lr}" \
  --eval_grid_t "${eval_grid_t}" \
  --eval_grid_s "${eval_grid_s}" \
  --intensity_grid_t "${intensity_grid_t}" \
  --intensity_grid_s "${intensity_grid_s}" \
  "$@"
