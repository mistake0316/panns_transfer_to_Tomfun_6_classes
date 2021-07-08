#!bin/bash
TRAIN_DATASET_DIR="./train_transfered"

WORKSPACE="."

PRETRAINED_CHECKPOINT_PATH="./Cnn14_mAP=0.431.pth"
AUGMENTATION="mixup" # one of ["none", "mixup"]
# Note: In TBrain Case, "none" is better than "mixup"

# # Train only one FOLD
# CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir=$TRAIN_DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation=$AUGMENTATION --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=600 --cuda --suffix="_train"

Train Ten FOLDs
for FOLD in 1 2 3 4 5 6 7 8 9 10
do
  CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir=$TRAIN_DATASET_DIR --workspace=$WORKSPACE --holdout_fold=$FOLD --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation=$AUGMENTATION --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=600 --cuda --suffix="_train"
done