#!/bin/bash
# Set up the data folder
IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
DATA_YAML="scripts/3d/train/grounding_subset.yaml" # e.g exp.yaml

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
alias python=python3

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="occ3dllm-LLaVA-Qwen-video-mm-only-vg"
PREV_STAGE_CHECKPOINT="ckpt/occ3dllm-LLaVA-Qwen-video-mm-only-v2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


NUM_GPUS=4
BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE/NUM_GPUS))

export CUDA_VISIBLE_DEVICES=1,2,3,5
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port 42001 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --lora_enable False \
    --embodiedscan_folder data/embodiedscan/ \
    --mm_tunable_parts="" \
    --mm_vision_tower_lr=2e-6 \
    --learning_rate 1e-5 \
    --vision_tower "google/siglip-so400m-patch14-384" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints  "(1x1)" \
    --mm_patch_merge_type spatial_unpad \
    --fp16 False \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./ckpt/$MID_RUN_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 12288 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy uniform \
    --frames_upbound 32 \

exit 0;