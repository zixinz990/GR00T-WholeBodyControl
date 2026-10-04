# Keep every cache inside this repo (nothing written to $HOME or system dirs)
REPO=/home/zixin/Dev/corl_loco_manipulation/GR00T-WholeBodyControl
export UV_CACHE_DIR=$REPO/.cache/uv
export UV_PYTHON_DOWNLOADS=never
export PIP_CACHE_DIR=$REPO/.cache/pip
export HF_HOME=$REPO/.cache/huggingface
export XDG_CACHE_HOME=$REPO/.cache
export MPLCONFIGDIR=$REPO/.cache/mpl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
