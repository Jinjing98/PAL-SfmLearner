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

 


CUDA_VISIBLE_DEVICES=0 python \
/mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/train_endodac.py \
--data_path /mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/ \
--num_workers 2 \
--num_epochs 30 \
--num_epochs 20 \
--batch_size 8 \
--log_frequency 200 \
--disparity_smoothness 0.01 \
--reconstruction_constraint 0.2 \
--reflec_constraint 0.2 \
--reprojection_constraint 1 \
--compute_metrics \
--val_full_eval \
--train_data_file d6_kf2.txt \
--val_data_file d6_kf2.txt \
--train_data_file train_files.txt \
--val_data_file val_files.txt \
--test_data_file test_files.txt \
--explicit_bias_init_6d9d \
--rot_representation 6D \
--rot_representation 9D \
--rot_representation angle_axis \
--backbone_size base \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac \
--pretrained_path /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/depthanything \
--exp_suffix full_endodacB_6D_baseline \
--exp_suffix full_endodacB_9D_baseline \
--exp_suffix full_endodacB_angleaxis_GTrot \
--exp_suffix full_endodacB_angleaxis_baseline_defaultPredictK \
--exp_suffix full_endodacB_angleaxis_optmizedK \
--learn_intrinsics True \
--of_samples \
--of_samples_num 16 \
--train_data_file d6_kf2.txt \
--val_data_file d6_kf2.txt \
--save_frequency 1000 \
--log_frequency 1 \
--num_epochs 20 \
--batch_size 2 \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac_dbg \


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