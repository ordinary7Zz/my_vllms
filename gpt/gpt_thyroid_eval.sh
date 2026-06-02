#!/bin/bash

# GPT Thyroid Classification Evaluation Script
# This script evaluates GPT-4o or other OpenAI vision models on thyroid nodule classification

# Configuration
MODEL="gpt-5.5"  # Choose: gpt-4-vision-preview, gpt-4o, gpt-4o-mini, gpt-5.5, gemini-3.5-flash, gemini-3.1-pro
IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images"
LABEL_JSON="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/sample_test_label.json"
OUTPUT_JSON="./outputs/gpt_thyroid_preds.json"
THRESHOLD=0.5
BOOTSTRAP_SAMPLES=2000

# OpenAI-compatible / Poe API key and endpoint
API_BASE_URL="${POE_API_BASE_URL:-${OPENAI_BASE_URL:-https://api.poe.com/v1}}"
API_KEY_FILE="sk-your-api-key-here"  # 在这里填写脚本内默认值
API_KEY="${POE_API_KEY:-${OPENAI_API_KEY:-$API_KEY_FILE}}"

# 确保API密钥已设置
if [ -z "$API_KEY" ] || [ "$API_KEY" = "sk-your-api-key-here" ]; then
    echo "Error: API key not configured"
    echo "可通过环境变量设置: export POE_API_KEY='xxxxx'"
    echo "或 export OPENAI_API_KEY='xxxxx'"
    exit 1
fi

# Run the evaluation
python gpt_thyroid_binary_eval.py \
    --image_dir "$IMAGE_DIR" \
    --label_json "$LABEL_JSON" \
    --out_json "$OUTPUT_JSON" \
    --model "$MODEL" \
    --base_url "$API_BASE_URL" \
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
