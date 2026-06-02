#!/bin/bash

# GPT Thyroid Classification Evaluation Script
# This script evaluates GPT-4o or other OpenAI vision models on thyroid nodule classification

# Configuration
MODEL="gpt-4o"  # Choose: gpt-4-vision-preview, gpt-4o, gpt-4o-mini
IMAGE_DIR="/path/to/image/directory"
LABEL_JSON="/path/to/label.json"
OUTPUT_JSON="./gpt_thyroid_preds.json"
THRESHOLD=0.5
BOOTSTRAP_SAMPLES=2000

# OpenAI API Key (直接写在这里，或从环境变量读取)
OPENAI_API_KEY="sk-***"  # 替换为你的实际API密钥

# 确保API密钥已设置
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-api-key-here" ]; then
    echo "Error: OPENAI_API_KEY not configured"
    echo "请编辑此脚本，将 'sk-your-api-key-here' 替换为你的实际OpenAI API密钥"
    echo "或通过环境变量设置: export OPENAI_API_KEY='sk-xxxxx'"
    exit 1
fi

# Run the evaluation
python gpt_thyroid_binary_eval.py \
    --image_dir "$IMAGE_DIR" \
    --label_json "$LABEL_JSON" \
    --out_json "$OUTPUT_JSON" \
    --model "$MODEL" \
    --threshold "$THRESHOLD" \
    --ci_bootstrap "$BOOTSTRAP_SAMPLES" \
    --ci_alpha 0.95 \
    --max_retries 3 \
    --retry_delay 2.0

# Optional: Run on a small subset first for testing
# python gpt_thyroid_binary_eval.py \
#     --image_dir "$IMAGE_DIR" \
#     --label_json "$LABEL_JSON" \
#     --out_json "$OUTPUT_JSON" \
#     --model "$MODEL" \
#     --limit 10
