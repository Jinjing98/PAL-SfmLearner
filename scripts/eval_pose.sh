#!/bin/bash

#SBATCH --job-name=end_pose
#SBATCH --gpus=rtxa5000:1   # rtxa5000 p6000 rtx6000 a100 v100 # monst3r requires 48GB each, only a100 supports
#SBATCH --nodes=1  # several gpus on one node
#SBATCH --ntasks-per-node=1 #used for multi gpu training
#SBATCH --mem=48G #64G #35G#25G  # 20G may cause bus error?   # mem * num_GPUS
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=8 #8 #4   #num works4 can not be too big;
#SBATCH --mail-user=xu.jinjing@uniklinikum-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/mnt/ceph/tco/TCO-Staff/Homes/jinjing/slurm_out/%j.err
#SBATCH --output=/mnt/ceph/tco/TCO-Staff/Homes/jinjing/slurm_out/%j.out
 
CUDA_VISIBLE_DEVICES=0 python /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/evaluate_pose.py \
--batch_size 1 \
--dataset endovis \
--data_path /mnt/cluster/workspaces/jinjingxu/SCARED_Images_Resized/ \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/MVP3R/baselines/DARES/af_sfmlearner_weights \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_again_1209_1324/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_with_adjustNet_6D_1209_1316/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_with_adjustNet_6D_gt_rot_learnK_1209_1319/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_PABA_with_adjustNet_6D_gt_rot_1209_1317/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_monov2_like_1205_1257/models/weights_29 \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/iidsfm/d6_kf2_IID_baseline_again_1210_1146/models/weights_12 \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/UniSfMLearner/submodule/Endo_FASt3r/dares_weights/best_weights \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_MIA \
--load_weights_folder /mnt/cluster/workspaces/jinjingxu/proj/PAL-SfmLearner/weights/af_pretrained/Model_trained_end_to_end \
--test_data_file d6_kf2.txt \
--test_data_file test_files_sequence2.txt \
--test_data_file test_files_sequence1.txt \
--rot_representation 6D \
--load_gt_from_npz \
--rot_representation angle_axis \
--load_weights_folder /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/endodac/DA3_testset/\
full_endodacB_angleaxis_baseline_OFrawSup_depthDA3_depth2disp_extraLoraOnAttnQK0TO11_wMultiScaleD_seqInputs_0115_1521/models/weights_19 \
# --track_lengths 411 \
# --load_gt_from_npz \

# traj1
# --track_lengths 411 \

# --save_poses_root /mnt/nct-zfs/TCO-Test/jinjingxu/exps/train/mvp3r/results/unisfm/eval_pose/ \

# --pose_model_type separate_resnet \
