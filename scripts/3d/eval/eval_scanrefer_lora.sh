#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE="results/scanrefer/test_occ_level1_vg_v3.jsonl"


CUDA_VISIBLE_DEVICES=2 python3 llava/eval/model_scanrefer.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 1 \
    --question-file data/processed/scanrefer_vg_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --overwrite_cfg true \
    --lora_path ./ckpt/occ3dllm-LLaVA-Qwen-level-1-vg-v3/checkpoint-7000/

python llava/eval/eval_scanrefer.py --input-file $ANWSER_FILE
