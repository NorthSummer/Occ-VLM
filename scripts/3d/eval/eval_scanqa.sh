#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE="results/scanqa/test_occvlm_video_deact_0.jsonl"


CUDA_VISIBLE_DEVICES=3 python3 llava/eval/model_scanqa.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 1 \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --question-file data/processed/scanqa_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true

python llava/eval/eval_scanqa.py --input-file $ANWSER_FILE