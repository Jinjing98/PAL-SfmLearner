#!/bin/bash

#SBATCH --job-name=end_pose
#SBATCH --gpus=a100:1   # rtxa5000 p6000 rtx6000 a100 v100 # monst3r requires 48GB each, only a100 supports
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
--depth_model_type endodac --pretrained_path /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/depthanything \
--da3_depth_regression_target depth2disp --depth_model_type depthanything3 --pretrained_path depth-anything/da3-base \
--da3_depth_regression_target disp --depth_model_type depthanything3 --pretrained_path depth-anything/da3-base \
--pose_model_type da3_internal \
--pose_model_type separate_resnet \
--k_model_type da3_internal \
--k_model_type mlp_with_pn_bottleneck_ipt \
--exp_suffix full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_disp \
--of_samples \
--of_samples_num 16 \
--of_samples_num 8 \
--save_frequency 1000 \
--log_frequency 1 \
--num_epochs 20 \
--batch_size 2 \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac_dbg \
--train_data_file test_files.txt \
--val_data_file test_files.txt test_files_sequence1_val.txt \

# --use_perframe_gt_K \
# --learn_intrinsics \
# --enable_seq_inputs \


# setup for endodac net
# --depth_model_type endodac --of_supervised_with_which outputs_refined --pose_model_type separate_resnet --da3_depth_regression_target disp --k_model_type mlp_with_pn_bottleneck_ipt --warm_up_step 5000 --pretrained_path /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/depthanything \

# Baseline EndoDAC with OFrawSup
#125433 full_endodacB_angleaxis_baseline_OFrawSup
# adjust_net AF help?
#125435 full_endodacB_angleaxis_baseline_OFrawSup_afAdjustNet
# GT K help?
#125437 full_endodacB_angleaxis_baseline_OFrawSup_gtK
# RAFT flow help?
#125436 full_endodacB_angleaxis_baseline_OFrawSup_ofRaft
# DA3_alone(single_scale; depth2disp) help?
#125438 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_raw_depth
# DA3_alone(single_scale; disp) help?
#125440 full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_disp

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