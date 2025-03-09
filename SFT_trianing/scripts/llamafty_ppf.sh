#!/bin/bash
# set -x;

GPU_IDS="0,1,2,3"

PER_DEV_BS=1
GRAD_ACC=4
LR=5e-6
EPOCHS=2.0
AUXILARY_SFT_COEF="0.2"

NUM_GPUS=$(echo $GPU_IDS | tr ',' ' ' | wc -w)
TOTAL_BS=$((NUM_GPUS * PER_DEV_BS * GRAD_ACC))

echo "Training configuration: GPU count=${NUM_GPUS}, \
Total batch size=${TOTAL_BS}, Per device batch size=${PER_DEV_BS}, \
Gradient accumulation=${GRAD_ACC}, Learning rate=${LR}, Epochs=${EPOCHS}"

MODEL_PATH="models/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR="sft-stage1_tbs${TOTAL_BS}_lr${LR}_ep${EPOCHS}"

RUN_NAME="coa-sft-stage1_tbs${TOTAL_BS}_lr${LR}_ep${EPOCHS}"
if [ ! -z "${AUXILARY_SFT_COEF}" ]; then
    RUN_NAME="${RUN_NAME}_aux${AUXILARY_SFT_COEF}"
    OUTPUT_DIR="${OUTPUT_DIR}_aux${AUXILARY_SFT_COEF}"
    AUXILARY_ARG="--auxilary_sft_coef_in_pl ${AUXILARY_SFT_COEF}"
else
    AUXILARY_ARG=""
fi

deepspeed --include localhost:${GPU_IDS} src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage rm \
    --model_name_or_path ${MODEL_PATH}  \
    --do_train \
    --dataset_dir data \
    --dataset partial_preference_coa \
    --template deepseek3_add-think_ppf \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size ${PER_DEV_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --plot_loss \
    --bf16 \
    --flash_attn fa2 \
    --cutoff_len 6144 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --enable_partial_perference_learning \
    ${AUXILARY_ARG}