#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

CKPT="./ckpt/$1"
ANWSER_FILE="results/sqa3d/test_occ_level3_qa.jsonl"
    

CUDA_VISIBLE_DEVICES=6 python3 llava/eval/model_sqa3d.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 1 \
    --question-file data/processed/sqa3d_test_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --overwrite_cfg true \
    --lora_path ./ckpt/occ3dllm-LLaVA-Qwen-level-1-qa/checkpoint-4000

python llava/eval/eval_sqa3d.py --input-file $ANWSER_FILE
