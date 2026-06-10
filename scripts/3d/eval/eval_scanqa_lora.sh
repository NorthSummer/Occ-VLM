#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

CKPT="./ckpt/$1"
ANWSER_FILE="results/scanqa/test_occ_level3_qa.jsonl"


CUDA_VISIBLE_DEVICES=5 python3 llava/eval/model_scanqa.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 1 \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --question-file data/processed/scanqa_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true \
    --lora_path ./ckpt/occ3dllm-LLaVA-Qwen-level-1-qa/checkpoint-4000

python llava/eval/eval_scanqa.py --input-file $ANWSER_FILE