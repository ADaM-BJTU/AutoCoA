GPU_IDS="0,1,2,3"

PER_DEV_BS=1
GRAD_ACC=8
LR=5e-5
EPOCHS=2.0

NUM_GPUS=$(echo $GPU_IDS | tr ',' ' ' | wc -w)
TOTAL_BS=$((NUM_GPUS * PER_DEV_BS * GRAD_ACC))

echo "Training configuration: GPU count=${NUM_GPUS}, \
Total batch size=${TOTAL_BS}, Per device batch size=${PER_DEV_BS}, \
Gradient accumulation=${GRAD_ACC}, Learning rate=${LR}, Epochs=${EPOCHS}"

MODEL_PATH="models/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR="sft-stage1_tbs${TOTAL_BS}_lr${LR}_ep${EPOCHS}"

EXP_NAME="coa-sft-stage1_2"

RUN_NAME="${EXP_NAME}_tbs${TOTAL_BS}_lr${LR}_ep${EPOCHS}"

deepspeed --include localhost:${GPU_IDS} src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --model_name_or_path ${MODEL_PATH}  \
    --do_train \
    --dataset_dir data \
    --dataset chain_of_action \
    --template deepseek3_add-think \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size ${PER_DEV_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --plot_loss \
    --bf16 \
    --flash_attn fa2 \
    --cutoff_len 8192 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --save_only_model \
    # --ignore_observation # add this flag to mask the observation content from external feedback