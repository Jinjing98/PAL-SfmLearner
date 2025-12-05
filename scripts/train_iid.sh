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
--log_dir /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm \
--data_path /mnt/nct-zfs/TCO-All/SharedDatasets/SCARED_Images_Resized/ \
--of_samples --of_samples_num 16 --save_frequency 10000 --log_frequency 200 --num_epochs 50000 \
