#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false


CKPT="./ckpt/$1"
ANWSER_FILE="results/scan2cap/test_occ_level1_v1.jsonl"

# example: sh scripts/3d/eval/eval_scan2cap_lora.sh $ckpt_name uniform 32
CUDA_VISIBLE_DEVICES=6 python3 llava/eval/model_scan2cap.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 1 \
    --question-file data/processed/scan2cap_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --overwrite_cfg true \
    --lora_path ./ckpt/occ3dllm-LLaVA-Qwen-level-1/checkpoint-1800

python llava/eval/eval_scan2cap.py --input-file $ANWSER_FILE