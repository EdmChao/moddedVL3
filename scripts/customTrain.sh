#!/bin/bash



echo "starting training"

mkdir -p $HOME/lib
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 $HOME/lib/libcuda.so
echo $HOME
echo $LD_LIBRARY_PATH

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
# ARG_NPROC_PER_NODE=${2:-8}
ARG_NPROC_PER_NODE=${2:-1}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

RUN_NAME="stage_1"
DATA_DIR="/content/moddedVL3/avaData"
OUTPUT_DIR="../models"

mkdir -p "$OUTPUT_DIR"

DEEPSPEED_CONFIG="/content/moddedVL3/scripts/zero1.json"
TRAIN_SCRIPT="/content/moddedVL3/videollama3/train.py"
export PYTHONPATH="/content/moddedVL3:$PYTHONPATH"
export FLASH_ATTENTION_FORCE_DISABLED=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export DISABLE_FLASH_ATTN=1
# This command uses DeepSpeed for distributed training and LoRA for memory-efficient finetuning.
deepspeed --launcher pytorch --master_port=29500 --num_gpus=1 "$TRAIN_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_type videollama3_qwen2 \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path "${DATA_DIR}/testingConversion.jsonl" \
    --data_folder ${DATA_DIR} \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 16 \
    --model_max_length 2048 \
    --mm_max_length 10240 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --mm_projector_lr 1e-3 \
    --vision_encoder_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --report_to tensorboard \
    --run_name $RUN_NAME
