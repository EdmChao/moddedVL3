#!/bin/bash

# Blind Navigation Fine-tuning Script for VideoLLaMA3
# Usage: ./train_blind_navigation.sh [num_samples] [model_size] [platform]

set -e

# Parse arguments
NUM_SAMPLES=${1:-1000}  # Start with manageable subset
# NUM_SAMPLES=${1:-56281} # full dataset
MODEL_SIZE=${2:-"2b"}   # 2b or 7b
PLATFORM=${3:-"cloud"}  # local, colab, or cloud
EPOCHS=3

# Configuration
TRAINING_DATA_DIR="/home/echao8/borgstore/avaData"
OUTPUT_DIR=$"../VITProjectorTest1000im3e"
RUN_NAME="blind_nav_$(date +%Y%m%d_%H%M%S)"

echo "=== VideoLLaMA3 Blind Navigation Training ==="
echo "Samples: $NUM_SAMPLES"
echo "Model size: $MODEL_SIZE"
echo "Platform: $PLATFORM"
echo "Output directory: $OUTPUT_DIR"
echo "============================================="



#Set platform-specific configurations
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
    #config for 2B
    BATCH_SIZE=2
    GRAD_ACCUM=16
    WORKERS=1
    MAX_FRAMES=8
    MAX_LENGTH=1024

    #config for 7B
    # BATCH_SIZE=2
    # GRAD_ACCUM=8
    # WORKERS=1
    # MAX_FRAMES=8
    # MAX_LENGTH=1024


else
    echo "Unknown platform: $PLATFORM. Use 'local', 'colab', or 'cloud'"
    exit 1
fi

#Set model configuration
if [ "$MODEL_SIZE" == "2b" ]; then
    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
    VISION_ENCODER="DAMO-NLP-SG/SigLIP-NaViT"
    PROJECTOR_TYPE="mlp2x_gelu"
    LR=1e-6
    MM_PROJECTOR_LR=1e-6
    VISION_LR=1e-6
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



echo "Training configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Max frames: $MAX_FRAMES"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"

echo "Step 2: Starting fine-tuning..."

export PYTHONPATH="/home/scratch/echao8/moddedVL3:$PYTHONPATH"
python /home/scratch/echao8/moddedVL3/videollama3/train.py \
    --deepspeed "/home/scratch/echao8/moddedVL3/scripts/zero2.json" \
    --model_type videollama3_qwen2 \
    --model_path "$MODEL_PATH" \
    --vision_encoder "$VISION_ENCODER" \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --data_path "${TRAINING_DATA_DIR}/convertedAnnotations1000.jsonl" \
    --data_folder "$TRAINING_DATA_DIR" \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames "$MAX_FRAMES" \
    --model_max_length "$MAX_LENGTH" \
    --mm_max_length $((MAX_LENGTH / 2)) \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate "$LR" \
    --llm_lr 0 \
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