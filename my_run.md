# 项目运行说明

## 一、当前项目用途

- 使用多模态模型对甲状腺超声图像做二分类
- 标签定义：`0 = benign`，`1 = malignant`
- 当前支持模型：Llama-3.2、MedGemma、Qwen3-VL
- 当前脚本会输出：每张图的预测 CSV + 总体评估指标 + 95% bootstrap 置信区间

## 二、运行前准备

### 1. 进入项目根目录

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms
```

### 2. 准备 Python 环境

需要保证环境中至少有这些包：

- `torch`
- `transformers`
- `pillow`
- `tqdm`
- `scikit-learn`
- `numpy`
- `modelscope`

### 3. 准备数据

需要：

- 图像目录 `image_dir`
- 标签文件 `label_json`

当前脚本默认使用过的数据示例：

```bash
image_dir=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images
label_json=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json
```

### 4. 标签文件格式

`label_json` 必须是一个 `list`，元素格式如下：

```json
[
  {"filename": "xxx.png", "malignancy": 0},
  {"filename": "yyy.png", "malignancy": 1}
]
```

## 三、模型准备

### 1. Llama-3.2

下载命令：

```bash
python download_llama.py
```

当前仓库根目录中的文件 `Llama-3.2-11B-Vision-Instruct` 记录了模型实际路径：

```text
/root/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-11B-Vision-Instruct
```

### 2. Qwen3-VL

下载命令：

```bash
python download_qwen3_vl_8b_instruct_modelscope.py
```

下载脚本默认目标目录：

```text
/mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct
```

### 3. MedGemma

仓库里没有下载脚本。

当前评估脚本示例使用的模型目录是：

```text
/mnt/wangbd8/workspace/medgemma-4b-it
```

## 四、直接运行命令

### 1. 运行 Llama-3.2

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/llama-3.2

python llama_thyroid_inference.py \
  --model_dir /root/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-11B-Vision-Instruct \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv llama_DDTI_preds.csv \
  --dtype fp16 \
  --threshold 0.5 \
  --ci_bootstrap 2000 \
  --ci_alpha 0.95 \
  --ci_seed 42
```

### 2. 运行 MedGemma

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/medgemma

python medgemma_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/medgemma-4b-it \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv medgemma_DDTI_Classification_all_preds.csv \
  --dtype bf16 \
  --threshold 0.5 \
  --ci_bootstrap 2000 \
  --ci_alpha 0.95 \
  --ci_seed 42
```

### 3. 运行 Qwen3-VL

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/qwen3

python qwen3_vl_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv qwen3_vl_DDTI_Classification_all_preds.csv \
  --dtype bf16 \
  --threshold 0.5 \
  --ci_bootstrap 2000 \
  --ci_alpha 0.95 \
  --ci_seed 42
```

## 五、微调命令示例

以下训练命令按单张 4090 24GB 显卡做了较快的默认配置：优先使用空闲卡，增大 batch size，减少梯度累积，并开启多进程数据加载。

### 1. 微调 MedGemma

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/medgemma

CUDA_VISIBLE_DEVICES=1 python medgemma_thyroid_binary_train.py \
  --model_dir /mnt/wangbd8/workspace/medgemma-4b-it \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/train/images \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/images \
  --train_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/train/dataset_3_train_label.json \
  --test_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/dataset_3_test_label.json \
  --output_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/vllms/medgemma/medgemma_dataset_3_lora \
  --dtype bf16 \
  --epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --learning_rate 2e-4 \
  --load_best_model_at_end

CUDA_VISIBLE_DEVICES=1 python medgemma_thyroid_binary_train.py \
  --model_dir /mnt/wangbd8/workspace/medgemma-4b-it \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/images \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/images \
  --train_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/thyroidxl_train_labels_sample.json \
  --test_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/thyroidxl_test_labels_sample.json \
  --output_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/vllms/medgemma/medgemma_sample_lora \
  --dtype bf16 \
  --epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --learning_rate 2e-4 \
  --load_best_model_at_end
```

### 2. 评估微调后的 MedGemma

```bash
python medgemma_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/medgemma-4b-it \
  --adapter_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/vllms/medgemma/medgemma_dataset_3_lora \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3/test/dataset_3_test_label.json \
  --out_csv medgemma_dataset_3_ft_preds.csv \
  --dtype bf16
```

### 3. 微调 Qwen3-VL

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/qwen3

CUDA_VISIBLE_DEVICES=2 python qwen3_vl_thyroid_binary_train.py \
  --model_dir /mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/images \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images \
  --train_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/TN5K_train_label.json \
  --test_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json \
  --output_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/vllms/qwen3/qwen3_TN5K_lora \
  --dtype bf16 \
  --use_qlora \
  --attn_impl sdpa \
  --epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --learning_rate 2e-4 \
  --load_best_model_at_end


CUDA_VISIBLE_DEVICES=2 python qwen3_vl_thyroid_binary_train.py \
  --model_dir /mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct \
  --train_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/images \
  --test_image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/images \
  --train_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/thyroidxl_train_labels_sample.json \
  --test_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/sample/thyroidxl_test_labels_sample.json \
  --output_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Agent/vllms/qwen3/qwen3_sample_lora \
  --dtype bf16 \
  --use_qlora \
  --attn_impl sdpa \
  --epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_workers 4 \
  --learning_rate 2e-4 \
  --load_best_model_at_end
```

### 4. 评估微调后的 Qwen3-VL

```bash
python qwen3_vl_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct \
  --adapter_dir /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/qwen3/qwen3_tn3k_lora \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/images \
  --label_json /mnt/wangbd8/workspace/ThyroidAgent/Classification_Models/my_vllms/my_json/tn3k_test_label.json \
  --out_csv qwen3_tn3k_ft_preds.csv \
  --dtype bf16
```

## 六、快速测试命令

只跑前 10 条样本：

### 1. Llama-3.2

```bash
python llama_thyroid_inference.py \
  --model_dir /root/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-11B-Vision-Instruct \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv llama_test_preds.csv \
  --dtype fp16 \
  --limit 10
```

### 2. MedGemma

```bash
python medgemma_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/medgemma-4b-it \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv medgemma_test_preds.csv \
  --dtype bf16 \
  --limit 10
```

### 3. Qwen3-VL

```bash
python qwen3_vl_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/images \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all/DDTI_Classification_test_label.json \
  --out_csv qwen3_test_preds.csv \
  --dtype bf16 \
  --limit 10
```

## 七、输出结果说明

### 1. 控制台会打印

- `AUROC`
- `AUPRC`
- `Accuracy`
- `F1`
- `Sensitivity`
- `Specificity`
- 每个指标对应的 `95% CI`
- 混淆矩阵 `tn fp fn tp`

### 2. `out_csv` 会保存每张图的结果

字段包括：

- `filename`
- `gt_malignant`
- `p_malignant`
- `pred_malignant`
- `logp_0`
- `logp_1`
- `cand0`
- `cand1`

## 八、参数说明

- `--model_dir`：模型本地目录
- `--image_dir`：图像目录
- `--label_json`：标签 json 文件
- `--out_csv`：输出预测结果 CSV
- `--dtype`：推理精度，可选 `bf16 / fp16 / fp32`
- `--threshold`：二分类阈值，默认 `0.5`
- `--limit`：只跑前 N 个样本，默认 `-1` 表示全部
- `--ci_bootstrap`：bootstrap 重采样次数，默认 `2000`
- `--ci_alpha`：置信水平，默认 `0.95`
- `--ci_seed`：随机种子，默认 `42`
- Qwen3-VL 额外支持：
  - `--attn_impl auto`
  - `--attn_impl flash_attention_2`
  - `--attn_impl sdpa`
  - `--attn_impl eager`

## 九、建议执行顺序

1. 先确认模型目录存在
2. 先用 `--limit 10` 做快速测试
3. 快速测试通过后再跑全量
4. 查看终端指标和输出 CSV
