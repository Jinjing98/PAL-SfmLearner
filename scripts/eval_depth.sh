#!/bin/bash

python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/evaluate_depth.py \
--data_path /mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/ \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_1205_1304/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_monov2_like_1205_1257/models/weights_29 \
--eval_split endovis \
--test_data_file d6_kf2_small.txt \
--test_data_file test_files.txt \
# --load_gt_from_npz \
# --test_data_file d6_kf2.txt \
# --save_pred_disps
# --ext_disp_to_eval /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg/d6_kf2_afstyle_baseline/disps_d6_kf2_split.npy