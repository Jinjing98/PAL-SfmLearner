#!/bin/bash

#AF-SFM E2E trained
#  Scaling ratios | med: 152.121 | std: 0.323
#            abs_rel      |      sq_rel      |        rmse      |    rmse_log      |          a1      |          a2      |          a3      | 
# mean:&       0.059      &       0.477      &       5.109      &       0.083      &       0.967      &       0.996      &       0.999      \\
# average inference time: 5.8 ms

#////////////wrongly set learn_intrinsics to True////////////////
#EndoDAC download_trained
#  Scaling ratios | med: 211.634 | std: 0.320
#            abs_rel      |      sq_rel      |        rmse      |    rmse_log      |          a1      |          a2      |          a3      | 
# mean:&       0.051      &       0.355      &       4.443      &       0.073      &       0.979      &       0.998      &       1.000      \\

#DA3 zero shot: resize to 224 280 the infer
#  Scaling ratios | med: 51.323 | std: 0.318
#            abs_rel      |      sq_rel      |        rmse      |    rmse_log      |          a1      |          a2      |          a3      | 
# mean:&       0.090      &       1.064      &       7.766      &       0.127      &       0.922      &       0.985      &       0.995      \\

#EndoDAC our_angleaxis_trained: warm up 20k
        #    abs_rel      |      sq_rel      |        rmse      |    rmse_log      |          a1      |          a2      |          a3      | 
# mean:&       0.054      &       0.404      &       4.770      &       0.077      &       0.975      &       0.997      &       0.999      \\

#EndoDAC our_angleaxis_trained: warm up 5k
        #    abs_rel      |      sq_rel      |        rmse      |    rmse_log      |          a1      |          a2      |          a3      | 
# mean:&       0.056      &       0.429      &       4.896      &       0.080      &       0.969      &       0.996      &       0.999      \\

#EndoDAC our_euler_trained: warm up 20k (also obvious better than 5K)
        #    abs_rel      |      sq_rel      |        rmse      |    rmse_log      |          a1      |          a2      |          a3      | 
# mean:&       0.055      &       0.404      &       4.753      &       0.077      &       0.974      &       0.997      &       0.999      \\

#EndoDAC our_9D_trained: warm up 20k/5k nan depth
#////////////wrongly set learn_intrinsics to True////////////////



# python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/third_party/EndoDAC/evaluate_depth.py \
python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/evaluate_depth_endodac.py \
--pretrained_path /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/depthanything \
--data_path /mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized/ \
--split endovis \
--eval_mono \
--model_type endodac --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/full_endodacB_angleaxis_baseline_1211_0028/models/weights_19 \
--model_type endodac --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/full_endodacB_euler_baseline_1212_1237/models/weights_19 \
--model_type endodac --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/full_endodacB_euler_baseline_warmUp5k_1215_1154/models/weights_19 \
--model_type endodac --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/full_endodacB_9D_baseline_1212_1642/models/weights_19 \
--model_type endodac --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/full_endodacB_9D_baseline_warmUp5k_1215_1153/models/weights_19 \
--model_type afsfm --load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_trained_end_to_end \
--model_type depthanything3 --load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/da3_placeholder \
--test_data_file d6_kf2_small.txt \
--test_data_file test_files.txt \
--test_data_file train_files.txt \
--save_pred_disps_online \
--compute_metadata_stats \
--visualize_depth \
--test_data_file d6_kf2_small.txt \
# --load_gt_from_npz \
# --visualize_depth \
# --save_folder place_folder \

# --save_per_frame_pred_root /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_trained_end_to_end/depth_predictions \
# --save_as_scaled \
# --user_given_scale_factor 1000 \

# cd /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner
# python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/evaluate_depth.py \
# --batch_size 8 \
# --data_path /mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized/ \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_afstyle_baseline_1205_1515/models/weights_29 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_1205_2103/models/weights_29 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_afstyle_refTgt_baseline_1205_2123/models/weights_29 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_with_adjustNet_1205_2216/models/weights_29 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_monov2_like_1205_1257/models/weights_29 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_1205_1304/models/weights_29 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg/d6_kf2_IID_baseline_again_1210_1132/models/weights_2 \
# --load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_again_1210_1146/models/weights_25 \
# --eval_split endovis \
# --load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_trained_end_to_end \
# --load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/DARES/best_weights \
# --load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_MIA \
# --test_data_file d6_kf2.txt \
# --test_data_file test_files.txt \
# --test_data_file d6_kf2_small.txt \
# # --load_gt_from_npz \
# # --use_metrics_py_batch \
# # --save_pred_disps
# # --test_data_file d6_kf2.txt \
# # --load_gt_from_npz \
# # --ext_disp_to_eval /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg/d6_kf2_afstyle_baseline/disps_d6_kf2_split.npy