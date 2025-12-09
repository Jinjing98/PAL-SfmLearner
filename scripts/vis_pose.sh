#!/bin/bash

#SBATCH --job-name=end_pose
#SBATCH --gpus=v100:1   # rtxa5000 p6000 rtx6000 a100 v100 # monst3r requires 48GB each, only a100 supports
#SBATCH --nodes=1  # several gpus on one node
#SBATCH --ntasks-per-node=1 #used for multi gpu training
#SBATCH --mem=48G #64G #35G#25G  # 20G may cause bus error?   # mem * num_GPUS
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8 #8 #4   #num works4 can not be too big;
#SBATCH --mail-user=xu.jinjing@uniklinikum-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/mnt/ceph/tco/TCO-Staff/Homes/jinjing/slurm_out/%j.err
#SBATCH --output=/mnt/ceph/tco/TCO-Staff/Homes/jinjing/slurm_out/%j.out
 
cd /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner
python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/vis_pose.py \
--dataset endovis \
--test_data_file test_files_sequence2.txt \
--test_data_file d6_kf2.txt \
--plot_xyz_rpy \
