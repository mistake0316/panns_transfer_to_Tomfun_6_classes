#!bin/bash
TRAIN_DATASET_DIR="./train_transfered"
PUBLIC_DATASET_DIR="./public_test_transfered"
PRIVATE_DATASET_DIR="./private_test_transfered"
WORKSPACE="."

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$TRAIN_DATASET_DIR --workspace=$WORKSPACE --suffix="_train"
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$PUBLIC_DATASET_DIR --workspace=$WORKSPACE --suffix="_public_test"
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$PRIVATE_DATASET_DIR --workspace=$WORKSPACE --suffix="_private_test"