#!/bin/bash

# ----- Shared Configuration -----
# TODO: Update these paths before running
CUDA_ID=0
FILTERED_CONCEPTS_DIR="path_to_save_filtered_concepts_dir"  # Directory to save filtered concept CSVs from step 2
SAVED_CROSS_MODAL_REP_DIR="path_to_save_cross_modal_representations_dir"  # Directory to cache teacher cross-modal representations (set to '' to disable caching)
PATH_TO_DICTIONARY="dictionaries/textgcd_tags_dictionary.csv"  # Path to the dictionary CSV file
EXP_NAME='train_SpectralGCD' # Name of the experiments group name for training runs (used for Weights & Biases logging)
EXP_ROOT='path_to_exp_root' # Root directory to save experiment outputs
USE_WANDB=false  # Set to false to disable wandb logging, else true to enable logging to Weights & Biases (wandb)

# Optional
WANDB_GROUP=${EXP_NAME}  # Weights & Biases group name (by default, set to the same as EXP_NAME, change if you want a different group name for logging)
PROJECT_NAME=''  # Weights & Biases project name
PATH_TO_KEY=''  # Path to the file containing the Weights & Biases API key

DATASET_NAME='cub'  # Dataset to use (cifar10, cifar100, imagenet_100, cub, scars, aircraft)

# Dataset-specific memax weights
declare -A MEMAX_WEIGHTS
MEMAX_WEIGHTS=(
    ["cub"]=2
    ["scars"]=2
    ["aircraft"]=1
    ["cifar10"]=1
    ["cifar100"]=4
    ["imagenet_100"]=1
)

# ----- Step 1: Save old/new class name splits -----

python -m utils.save_old_class_names \
    --dataset_name $DATASET_NAME \
    --use_ssb_splits

# ----- Step 2: Spectral filtering -----

python spectral_filtering.py \
    --dataset_name $DATASET_NAME \
    --batch_size 128 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --transform imagenet \
    --seed 0 \
    --use_torch_impl \
    --thresholding_eig 0.95 \
    --thresholding_concepts 0.99 \
    --cuda_dev $CUDA_ID \
    --path_to_filtered_concepts $FILTERED_CONCEPTS_DIR \
    --path_to_dictionary $PATH_TO_DICTIONARY \
    --exp_root $EXP_ROOT \
    --exp_id "${DATASET_NAME}_spectral_filtering_$(date +%Y%m%d_%H%M%S)"

# ----- Step 3: Train SpectralGCD -----


for seed in {0..2}
do
    exp_name="${DATASET_NAME}_${EXP_NAME}"
    exp_id="${exp_name}_$(date +%Y%m%d_%H%M%S)"

    python spectralgcd.py \
        --dataset_name $DATASET_NAME \
        --batch_size 128 \
        --epochs 200 \
        --num_workers 8 \
        --use_ssb_splits \
        --sup_weight 0.35 \
        --weight_decay 5e-5 \
        --transform imagenet \
        --lr 0.1 \
        --lr_backbone 0.005 \
        --warmup_teacher_temp 0.07 \
        --teacher_temp 0.04 \
        --warmup_teacher_temp_epochs 30 \
        --memax_weight ${MEMAX_WEIGHTS[$DATASET_NAME]} \
        --seed $seed \
        --cuda_dev $CUDA_ID \
        --exp_id $exp_id \
        --path_to_saved_cross_modal_representations $SAVED_CROSS_MODAL_REP_DIR \
        --path_to_filtered_concepts "${FILTERED_CONCEPTS_DIR}/${DATASET_NAME}_concepts.csv" \
        --project_name $PROJECT_NAME \
        --w_key_path $PATH_TO_KEY \
        --group_name $WANDB_GROUP \
        --exp_root $EXP_ROOT \
        --experiment_name "${DATASET_NAME}_${EXP_NAME}" \
        $( [ "$USE_WANDB" = "true" ] && echo "--use_wandb" )
done
