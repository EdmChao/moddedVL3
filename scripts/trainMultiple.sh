#!/bin/bash

# Blind Navigation Fine-tuning Script for VideoLLaMA3
# Usage: ./train_blind_navigation.sh [num_samples] [model_size] [platform]

set -e

# Parse arguments
NUM_SAMPLES=${1:-5000}  # Start with manageable subset
MODEL_SIZE=${2:-"2b"}   # 2b or 7b
PLATFORM=${3:-"cloud"}  # local, colab, or cloud

# Configuration
TRAINING_DATA_DIR="../avaData"
OUTPUT_DIR="../finetuneCheckpoint"
RUN_NAME="blind_nav_$(date +%Y%m%d_%H%M%S)"

echo "=== VideoLLaMA3 Blind Navigation Training ==="
echo "Samples: $NUM_SAMPLES"
echo "Model size: $MODEL_SIZE"
echo "Platform: $PLATFORM"
echo "Output directory: $OUTPUT_DIR"
echo "============================================="

# Step 1: Prepare training data
# echo "Step 1: Preparing training data..."
# python prepare_blind_navigation_data.py \
#     --annotation_json "data/annotations/split_train.json" \
#     --video_dir "videollama3/videos" \
#     --output_dir "$TRAINING_DATA_DIR" \
#     --num_samples "$NUM_SAMPLES" \
#     --create_symlinks

# Step 2: Set platform-specific configurations
if [ "$PLATFORM" == "local" ]; then
    echo "Configuring for local M3 Max training..."
    BATCH_SIZE=1
    GRAD_ACCUM=8
    WORKERS=2
    MAX_FRAMES=32
    MAX_LENGTH=4096
elif [ "$PLATFORM" == "colab" ]; then
    echo "Configuring for Google Colab..."
    BATCH_SIZE=2
    GRAD_ACCUM=4
    WORKERS=4
    MAX_FRAMES=64
    MAX_LENGTH=8192
elif [ "$PLATFORM" == "cloud" ]; then
    echo "Configuring for cloud GPU training..."
    BATCH_SIZE=2
    GRAD_ACCUM=2
    WORKERS=4
    MAX_FRAMES=16
    MAX_LENGTH=8192
else
    echo "Unknown platform: $PLATFORM. Use 'local', 'colab', or 'cloud'"
    exit 1
fi

# Step 3: Set model configuration
if [ "$MODEL_SIZE" == "2b" ]; then
    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
    VISION_ENCODER="DAMO-NLP-SG/SigLIP-NaViT"
    PROJECTOR_TYPE="mlp2x_gelu"
    LR=2e-5
    MM_PROJECTOR_LR=1e-4
    VISION_LR=1e-5
elif [ "$MODEL_SIZE" == "7b" ]; then
    MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    VISION_ENCODER="DAMO-NLP-SG/SigLIP-NaViT"
    PROJECTOR_TYPE="mlp2x_gelu"
    LR=1e-5
    MM_PROJECTOR_LR=5e-5
    VISION_LR=5e-6
    # Reduce batch size for 7B model
    BATCH_SIZE=$((BATCH_SIZE / 2))
    GRAD_ACCUM=$((GRAD_ACCUM * 2))
else
    echo "Unsupported model size: $MODEL_SIZE. Use '2b' or '7b'"
    exit 1
fi

# Calculate epochs based on dataset size
# if [ "$NUM_SAMPLES" -lt 1000 ]; then
#     EPOCHS=5
# elif [ "$NUM_SAMPLES" -lt 5000 ]; then
#     EPOCHS=3
# else
#     EPOCHS=1
# fi
EPOCHS=1

echo "Training configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Max frames: $MAX_FRAMES"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"

# Step 4: Run training
echo "Step 2: Starting fine-tuning..."

# Use MPS for M3 Max if available, otherwise CUDA
# export PYTORCH_ENABLE_MPS_FALLBACK=1

export PYTHONPATH="/home/scratch/echao8/moddedVL3:$PYTHONPATH"
python /home/scratch/echao8/moddedVL3/videollama3/train.py \
    --model_type videollama3_qwen2 \
    --model_path "$MODEL_PATH" \
    --vision_encoder "$VISION_ENCODER" \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --data_path "${TRAINING_DATA_DIR}/testingConversions.jsonl" \
    --data_folder "$TRAINING_DATA_DIR" \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames "$MAX_FRAMES" \
    --model_max_length "$MAX_LENGTH" \
    --mm_max_length $((MAX_LENGTH / 2)) \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate "$LR" \
    --mm_projector_lr "$MM_PROJECTOR_LR" \
    --vision_encoder_lr "$VISION_LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers "$WORKERS" \
    --report_to "none" \
    --run_name "$RUN_NAME" \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --remove_unused_columns False

echo "============================================="
echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To test your model:"
echo "  python test_single_video_model.py --model_path $OUTPUT_DIR --video_path [your_video.mp4]"
echo ""
echo "Training stats:"
echo "  Dataset size: $NUM_SAMPLES samples"
echo "  Training time estimate: $(( (NUM_SAMPLES * EPOCHS) / (BATCH_SIZE * GRAD_ACCUM * 60) )) minutes"
echo "============================================="