#!/bin/bash

# eval Depth (scene-wise) on the offical test split of scared (Aligh once per sequence)
# save_path will be constructed based on model name
# infact is call demo.py


# eval depth on scared test
cd /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/third_party/Endo3R
python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/evaluate_depth_endo3rStyle.py \
    --data_root /mnt/ceph/tco/TCO-Staff/Homes/jinjing/Datasets/SCARED_rectified/test/ \
    --data_type scared \
    --resolution 320 \
    --kf_every 1 \
    --endo3r_scale