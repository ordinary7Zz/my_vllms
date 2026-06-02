#!/bin/bash

# GPT Thyroid Classification Evaluation Script
# This script evaluates GPT-4o or other OpenAI vision models on thyroid nodule classification

# Configuration
MODEL="gpt-5.5"  # Choose: gpt-4-vision-preview, gpt-4o, gpt-4o-mini
IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images"
LABEL_JSON="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/sample_test_label.json"
OUTPUT_JSON="./gpt/outputs/gpt_thyroid_preds.json"
THRESHOLD=0.5
BOOTSTRAP_SAMPLES=2000

# OpenAI API Key（优先读取环境变量，其次使用脚本内配置）
OPENAI_API_KEY_FILE="sk-your-api-key-here"  # 在这里填写脚本内默认值
OPENAI_API_KEY="${OPENAI_API_KEY:-$OPENAI_API_KEY_FILE}"

# 确保API密钥已设置
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-api-key-here" ]; then
    echo "Error: OPENAI_API_KEY not configured"
    echo "可通过环境变量设置: export OPENAI_API_KEY='sk-xxxxx'"
    echo "或编辑此脚本，将 OPENAI_API_KEY_FILE 替换为你的实际OpenAI API密钥"
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
