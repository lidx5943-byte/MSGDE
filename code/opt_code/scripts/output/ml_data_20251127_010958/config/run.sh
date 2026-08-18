#!/bin/bash
# 实验: ml_data
# 时间: 2025-11-27 01:09:58

cd /srv/wzh/mm_eeg/opt_code/scripts
run_ml_convert.py --features dynamics_data.npy --trajectories trajectories.npy
