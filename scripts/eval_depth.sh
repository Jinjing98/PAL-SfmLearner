#!/bin/bash

python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/evaluate_depth.py \
--batch_size 8 \
--data_path /mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/ \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_afstyle_baseline_1205_1515/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_1205_2103/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_afstyle_refTgt_baseline_1205_2123/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_with_adjustNet_1205_2216/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_monov2_like_1205_1257/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_1205_1304/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg/d6_kf2_IID_baseline_again_1210_1132/models/weights_2 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_again_1210_1146/models/weights_25 \
--eval_split endovis \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/DARES/af_sfmlearner_weights \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_trained_end_to_end \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_MIA \
--load_gt_from_npz \
--test_data_file d6_kf2.txt \
--test_data_file test_files.txt \
# --test_data_file d6_kf2_small.txt \
# --use_metrics_py_batch \
# --save_pred_disps
# --test_data_file d6_kf2.txt \
# --load_gt_from_npz \
# --ext_disp_to_eval /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg/d6_kf2_afstyle_baseline/disps_d6_kf2_split.npy