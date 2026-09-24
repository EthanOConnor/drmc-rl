#!/bin/bash
# tf3090 launcher for experiment A (staged under /dev/shm/timing-contract; the root disk is full).
R=/dev/shm/timing-contract
export DRMC_FRAME_LIBRARY=$R/native/libdrmario_pool.so DRMARIO_POOL_LIB=$R/native/libdrmario_pool.so
export DRMARIO_REACH_LIB=$R/native/libdrm_reach_full.so PYTHONPATH=$R/code PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR=$R/tmp
export PATH=/home/ethan/dev/drmario/drmc-rl/.venv/bin:$PATH
mkdir -p $R/tmp
cd $R/code
exec "$@"
