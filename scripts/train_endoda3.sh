#!/bin/bash

#SBATCH --job-name=end_pose
#SBATCH --gpus=v100:1   # rtxa5000 p6000 rtx6000 a100 v100 # monst3r requires 48GB each, only a100 supports
#SBATCH --nodes=1  # several gpus on one node
#SBATCH --ntasks-per-node=1 #used for multi gpu training
#SBATCH --mem=48G #64G #35G#25G  # 20G may cause bus error?   # mem * num_GPUS
#SBATCH --time=46:00:00
#SBATCH --cpus-per-task=4 #8 #4   #num works4 can not be too big;
#SBATCH --mail-user=xu.jinjing@uniklinikum-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/mnt/nct-zfs/TCO-Test/jinjingxu/slurm_out/%j.err
#SBATCH --output=/mnt/nct-zfs/TCO-Test/jinjingxu/slurm_out/%j.out

 
# avoid pip -e da3
# export PYTHONPATH=$PYTHONPATH:/mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/third_party/depth_anything_3

# CUDA_VISIBLE_DEVICES=0 python \
# /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/networks/endo_da3.py
CUDA_VISIBLE_DEVICES=0 python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/train_endoda3.py \
--data_path /mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized/ \
--num_workers 2 \
--num_workers 0 \
--num_epochs 20 \
--batch_size 8 \
--log_frequency 200 \
--disparity_smoothness 0.01 \
--compute_depth_metrics \
--compute_pose_metrics \
--val_full_eval \
--train_data_file d6_kf2.txt \
--val_data_file d6_kf2.txt \
--train_data_file train_files.txt \
--val_data_file val_files.txt \
--val_data_file test_files.txt test_files_sequence1_val.txt test_files_sequence2_val.txt \
--test_data_file test_files.txt \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endoDA3 \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/DA3/ \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/DA3_testset/ \
--exp_suffix full_endodacB_6D_baseline \
--exp_suffix full_endodacB_angleaxis_GTrot \
--exp_suffix full_endodacB_9D_defInit_naiveMul_baseline \
--exp_suffix full_endodacB_quat_baseline \
--exp_suffix full_endodacB_quat_regressXYZ_baseline \
--exp_suffix full_endodacB_9D_baseline \
--exp_suffix full_endodacB_angleaxis_baseline_warmUp5k \
--exp_suffix full_endodacB_9D_baseline_warmUp5k \
--exp_suffix full_endodacB_euler_baseline_warmUp5k \
--exp_suffix full_endoDA3B_DepthOnly_angleaxis_baseline_LorawarmUp40k_woRes_SingleScale \
--exp_suffix full_endoDA3B_DepthPoseK_quanXYZW001_baseline_LorawarmUp40k_woRes_SingleScale \
--exp_suffix full_endoDA3B_DepthPoseK_angleaxis001_baseline_LorawarmUp40k_woRes_SingleScale_LearnIntrinsics \
--exp_suffix full_endoDA3B_DepthPoseK_angleaxis001_baseline_LorawarmUp40k_woRes_SingleScale_RAFT \
--exp_suffix full_endoDA3B_DepthK_angleaxis001_baseline_LorawarmUp40k_woRes_SingleScale_LearnIntrinsics \
--exp_suffix full_endoDA3B_DepthK_angleaxis001_baseline_LorawarmUp40k_woRes_SingleScale_LearnIntrinsics_fixMINDEPTH_ufzAllRaftLastOnly \
--warm_up_step 20000 \
--warm_up_step 5000 \
--warm_up_step 40000 \
--af_model_type adjust_net \
--af_model_type separate_resnet \
--of_supervised_with_which outputs_refined \
--of_supervised_with_which inputs_color \
--of_model_type raft \
--of_model_type separate_resnet \
--raft_trainable_modules convnormrelu layer1 layer2_0 \
--raft_trainable_modules all \
--use_raft_multi_iters \
--endoda3_model_config /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/networks/configs/endo-da3-all-wowrapper.yaml \
--da3_depth_regression_target depth2disp_v2 --depth_model_type depthanything3 --pretrained_path depth-anything/da3-base \
--da3_depth_regression_target disp --depth_model_type depthanything3 --pretrained_path depth-anything/da3-base \
--da3_depth_regression_target depth2disp --depth_model_type depthanything3 --pretrained_path depth-anything/da3-base \
--depth_model_type endodac --pretrained_path /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/depthanything \
--pose_model_type da3_internal \
--pose_model_type separate_resnet \
--k_model_type da3_internal \
--k_model_type mlp_with_pn_bottleneck_ipt \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_scratchHead \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_9D \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_euler \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_ofRaftEarlyLayers \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_GTPose5e04 \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2dispV2normSigmoid_wMultiScaleD \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2dispV2normSigmoid_woDispsmooth_wMultiScaleD \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_GTPose5e04 \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_GTPose5e03 \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_GTPoseRotOnly \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_lre05 \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_lre05DepthOnly \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_ofRaftLastIterOnly \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_ofRaftLastIterOnly_freezeOF \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_disp_wMultiScaleD_scratchHead_woScalingInDisp2DepthinTrn \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDA3Pretrainedfc_fov \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDA3Scratchfc_fov_bounded_linear_sigmoid \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_gtK \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDacStyle \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_transDataAug03 \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_trnFrameidDelta5 \
--train_frame_ids 0 -5 5 \
# --exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_transDataAug03_extraautomasking \
# --exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_seqInput \
# --of_samples \
# --of_samples_num 16 \
# --of_samples_num 8 \
# --save_frequency 1000 \
# --log_frequency 1 \
# --num_epochs 20 \
# --train_data_file test_files.txt \
# --train_data_file test_files_sequence1_val.txt \
# --train_data_file train_files.txt \
# --val_data_file test_files_sequence1_val.txt \
# --val_data_file test_files_sequence2_val.txt test_files_sequence1_val.txt test_files.txt \
# --val_data_file test_files.txt \
# --log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac_dbg \
# --train_frame_ids 0 -5 5 \
# # # # --batch_size 2 \

# # --use_perframe_gt_K \
# # --learn_intrinsics \
# # --enable_seq_inputs \


# setup for endodac net
# --depth_model_type endodac --of_supervised_with_which outputs_refined --pose_model_type separate_resnet --da3_depth_regression_target disp --k_model_type mlp_with_pn_bottleneck_ipt --warm_up_step 5000 --pretrained_path /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/depthanything \

# explict OF supervision?
# raw_disp; then skip scale in disp2depth

# trans based data augmentation
# 125942(aug on color and color_aug) 125938 (wrongly only on color_aug) full_endodacB_angleaxis_baseline_OFrawSup_transDataAug03
# auto masking furhter helps 
# 125943 full_endodacB_angleaxis_baseline_OFrawSup_transDataAug03_extraautomasking
# trn data trans_rot_aug
# 126003 full_endodacB_angleaxis_baseline_OFrawSup_trnFrameidDelta5

# can we have more consistent depth via seq input?
# 125822 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_seqInput

# can we involve learned k from DA3 to improve overral:
# 125821 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDA3

# scratch disp + DA3 work is limited due to too much nolinear?
# 125817 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_disp_wMultiScaleD_scratchHead_woScalingInDisp2DepthinTrn
# 125924 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDA3Pretrainedfc_fov
# 125926 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDA3Scratchfc_fov_bounded_linear_sigmoid
# 125927 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_gtK
# 125928 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_KDacStyle

# DA3 need smaller lr?
# 125741 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_lre05
# only depthnet lr is changed to 1e-5 other remain 1e-4
# 125762 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_lre05DepthOnly

#  use explict OF supervision?

# verify if RAFT helps or not: raft_multi_all is only compariable
# 125765 full_endodacB_angleaxis_baseline_OFrawSup_ofRaftLastIterOnly
# verify if RAFT need to be tuned or not
# 125767 full_endodacB_angleaxis_baseline_OFrawSup_ofRaftLastIterOnly_freezeOF

# gt_pose: upperbound. sensitive to trans scale?
# 125738 full_endodacB_angleaxis_baseline_OFrawSup_GTPose5e03
# 125739 full_endodacB_angleaxis_baseline_OFrawSup_GTPoseRotOnly

# avoid smooth disp as DA3 already smooth it.
# 125724 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2dispV2normSigmoid_woDispsmooth_wMultiScaleD
# later saturation of DA3 backbone?
# 125723 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2dispV2normSigmoid_wMultiScaleD

# 125708 full_endodacB_angleaxis_baseline_OFrawSup_GTPose5e04
# wrong 125648 full_endodacB_angleaxis_baseline_OFrawSup_GTPose5e04

# 125595 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_ofRaftEarlyLayers
# 125593 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_euler
# 125593 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_9D
# failed 125592 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_GTPose5e04
# 125590 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD_scratchHead
# 125591 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_disp_wMultiScaleD_scratchHead 

# DA3 with Multi-Scale Depth help?
#125485 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_wMultiScaleD
# Baseline EndoDAC with OFrawSup
#125452 125445 125433 full_endodacB_angleaxis_baseline_OFrawSup
# adjust_net AF help?
#125454 125447 125435 full_endodacB_angleaxis_baseline_OFrawSup_afAdjustNet
# GT K help?
#125453 125446 125437 full_endodacB_angleaxis_baseline_OFrawSup_gtK
# RAFT flow help?
#125459 125451 125436 full_endodacB_angleaxis_baseline_OFrawSup_ofRaft
# DA3_alone(single_scale; depth2disp) help?
#125455 125448 125438 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp
# DA3_alone(single_scale; disp) help?
#125458125450 125440 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_disp

#125086 full_endoDA3B_DepthOnly_angleaxis_baseline_LorawarmUp40k_woRes_SingleScale: is infact no lora:None
#125087 full_endoDA3B_DepthOnly_angleaxis_baseline_LorawarmUp40k_woRes_SingleScale: correcct lora fine tune



# --reproj_supervise_type color_warp \
# --reproj_supervise_type afstyle_color_warp \
# --reproj_supervise_type paba_color_warp \
# --reproj_supervise_type reprojection_color_warp \
# --reflec_constraint 0.0 \
# --trans_scale_factor 0.001 \
# --rot_scale_factor 0.001 \

# --exp_suffix angleaxis_use_gt_rot \
# --exp_suffix d6_kf2_PABA_with_adjustNet_gt_rot \
# --of_samples --train_data_file val_files.txt --val_data_file val_files.txt  --of_samples_num 16 --save_frequency 10000 --log_frequency 300 --num_epochs 500 --log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg \
# --reproj_supervise_type paba_color_warp \
# --exp_suffix d6_kf2_afsfm_like \

# 122093 d6_kf2_monov2_like: --reflec_constraint 0.0 --reproj_supervise_type color_warp  \
# 122111 d6_kf2_IID_baseline
# 122202 d6_kf2_afstyle_baseline

#122668  d6_kf2_PABA_with_adjustNet_6D