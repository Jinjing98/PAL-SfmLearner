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


cd /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner

CUDA_VISIBLE_DEVICES=0 python \
/mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/train.py \
--num_workers 2 \
--num_epochs 30 \
--batch_size 8 \
--log_frequency 200 \
--disparity_smoothness 0.01 \
--reconstruction_constraint 0.2 \
--reflec_constraint 0.2 \
--reprojection_constraint 1 \
--compute_metrics \
--train_data_file train_files.txt \
--val_data_file val_files.txt \
--test_data_file test_files.txt \
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm \
--data_path /mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/ \
--train_data_file d6_kf2.txt \
--val_data_file d6_kf2.txt \
--exp_suffix d6_kf2_IID_baseline \
--exp_suffix d6_kf2_afstyle_baseline \
--reflec_constraint 0.0 \
--reproj_supervise_type reprojection_color_warp \
--reproj_supervise_type color_warp \
--reproj_supervise_type afstyle_color_warp \
# --of_samples --train_data_file val_files.txt --val_data_file val_files.txt  --of_samples_num 16 --save_frequency 10000 --log_frequency 200 --num_epochs 50000 --log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm_dbg \
# --reproj_supervise_type paba_color_warp \
# --exp_suffix d6_kf2_afsfm_like \

# 122093 d6_kf2_monov2_like: --reflec_constraint 0.0 --reproj_supervise_type color_warp  \
# 122111 d6_kf2_IID_baseline
# 122202 d6_kf2_afstyle_baseline