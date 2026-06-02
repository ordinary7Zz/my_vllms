# GPT Vision Models for Thyroid Classification

本目录包含使用OpenAI兼容视觉接口（如 OpenAI / Poe）进行甲状腺超声恶性肿瘤分类的脚本。

## 功能特性

- ✅ 支持多个OpenAI兼容模型：`gpt-4o`、`gpt-4-vision-preview`、`gpt-4o-mini`、`gpt-5.5`、`gemini-3.5-flash`、`gemini-3.1-pro`
- ✅ 图像编码为base64，无需本地上传
- ✅ 结构化JSON响应解析，提取置信度分数
- ✅ 完整的分类指标计算（AUROC、AUPRC、Accuracy、F1、Sensitivity、Specificity）
- ✅ **Bootstrap置信区间**（95% CI）用于所有指标
- ✅ 指数退避重试机制处理速率限制
- ✅ JSON输出包含每个样本的预测、概率和模型推理

## 安装依赖

```bash
# 基础依赖
pip install openai>=1.0.0 numpy scikit-learn pillow tqdm

# 如果需要数据分析
pip install pandas matplotlib seaborn
```

## 配置API密钥

OpenAI API密钥配置方法：

如果你使用 Poe / 其他 OpenAI-compatible endpoint（例如 `gpt-5.5`、`gemini-3.5-flash`、`gemini-3.1-pro`），请额外设置：
```bash
export POE_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
export POE_API_BASE_URL='https://api.poe.com/v1'
```
### 方法1：环境变量（推荐）
```bash
export OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

### 方法2：命令行参数
```bash
python gpt_thyroid_binary_eval.py --api_key 'sk-xxxxxxxxxxxxxxx' ...
```

### 方法3：创建.env文件
```bash
# 创建 .env 文件
echo "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx" > .env

# Python中加载
from dotenv import load_dotenv
load_dotenv()
```

## 使用方法

### 基本用法

```bash
python gpt_thyroid_binary_eval.py \
    --image_dir /path/to/images \
    --label_json /path/to/labels.json \
    --out_json predictions.json \
    --model gpt-4o
```

### 完整参数说明

```bash
python gpt_thyroid_binary_eval.py \
    --image_dir /path/to/images                    # 图像目录 [必需]
    --label_json /path/to/labels.json              # 标签JSON文件 [必需]
    --out_json output.json                           # 输出JSON文件 (默认: gpt_thyroid_preds.json)
    --model gpt-4o                                 # 模型选择 (默认: gpt-4o)
    --threshold 0.5                                # 分类阈值 (默认: 0.5)
    --limit 100                                    # 仅评估前N个样本，用于测试 (默认: -1,全部)
    --ci_bootstrap 2000                            # Bootstrap重采样次数 (默认: 2000)
    --ci_alpha 0.95                                # 置信区间水平 (默认: 0.95)
    --max_retries 3                                # API调用最大重试次数 (默认: 3)
    --retry_delay 2.0                              # 重试延迟时间(秒) (默认: 2.0)
```

### 快速测试

先用小数据集测试：
```bash
python gpt_thyroid_binary_eval.py \
    --image_dir ./test_images \
    --label_json ./test_labels.json \
    --limit 5  # 仅处理5个样本
```

### 使用脚本

编辑`gpt_thyroid_eval.sh`配置参数，然后运行：
```bash
chmod +x gpt_thyroid_eval.sh
./gpt_thyroid_eval.sh
```

## 标签JSON格式

标签文件应为JSON列表，每条记录包含文件名和标签：

```json
[
    {
        "filename": "patient001_thyroid.jpg",
        "malignancy": 0
    },
    {
        "filename": "patient002_thyroid.jpg",
        "malignancy": 1
    }
]
```

- `filename`: 相对于`--image_dir`的图像文件路径
- `malignancy`: 标签，0=良性(benign)，1=恶性(malignant)

## 模型选择

| 模型 | 价格 | 性能 | 推荐用途 |
|------|------|------|---------|
| `gpt-4-vision-preview` | 高 | 最强 | 精确医学诊断 |
| `gpt-4o` | 中 | 很强 | 平衡选择 |
| `gpt-4o-mini` | 低 | 良好 | 快速原型/成本敏感 |
| `gpt-5.5` | 取决于平台 | 很强 | Poe 兼容接口 |
| `gemini-3.5-flash` | 取决于平台 | 快 | Poe 兼容接口 |
| `gemini-3.1-pro` | 取决于平台 | 很强 | Poe 兼容接口 |

## 输出文件

### JSON输出（sample records）

```json
[
  {
    "record_type": "sample",
    "image_file": "patient001.jpg",
    "image_name": "patient001.jpg",
    "filename": "patient001.jpg",
    "selected_model": "gpt-4o",
    "predicted_class": 0,
    "confidence": 0.857143,
    "prob_class_0": 0.857143,
    "prob_class_1": 0.142857,
    "true_label": 0,
    "parsed_prediction": 0,
    "parsed_confidence": 0.142857,
    "reasoning": "Model primarily identifies benign characteristics..."
  },
  {
    "record_type": "sample",
    "image_file": "patient002.jpg",
    "image_name": "patient002.jpg",
    "filename": "patient002.jpg",
    "selected_model": "gpt-4o",
    "predicted_class": 1,
    "confidence": 0.857143,
    "prob_class_0": 0.142857,
    "prob_class_1": 0.857143,
    "true_label": 1,
    "parsed_prediction": 1,
    "parsed_confidence": 0.857143,
    "reasoning": "Image shows suspicious features consistent with..."
  }
]
```

- `true_label`: 真实标签（0=良性，1=恶性）
- `prob_class_1`: 模型预测的恶性概率（0-1）
- `predicted_class`: 基于阈值的二分类预测（0或1）
- `parsed_prediction`: 模型JSON响应中的预测值
- `reasoning`: 模型提供的分析原因

### 控制台输出

```
================================================================================
GPT Thyroid Binary Classification Metrics (pos=malignant=1)
================================================================================
Model:             gpt-4o
Evaluated samples: 100
Missing files:     0
Bad images:        0
API errors:        0
Decision threshold: 0.5
--------------------------------------------------------------------------------
AUROC:             0.890123 (95% CI 0.862456-0.915789)
AUPRC:             0.875432 (95% CI 0.840123-0.901234)
Acc:               0.850000 (95% CI 0.810000-0.880000)
F1:                0.823529 (95% CI 0.780000-0.860000)
Sensitivity:       0.833333 (95% CI 0.760000-0.890000)
Specificity:       0.869565 (95% CI 0.820000-0.910000)
Confusion (tn fp fn tp): 60 9 8 23
--------------------------------------------------------------------------------
Saved per-sample predictions to: gpt_thyroid_preds.json
```

## Bootstrap置信区间说明

置信区间采用**Bootstrap重采样方法**计算：

1. **原理**：从数据中有放回地随机抽样N次（默认2000次），每次计算指标
2. **优势**：
   - 无需假设数据分布
   - 自动处理偏态分布
   - 对小样本更稳健
3. **解释**：95% CI表示有95%把握真实指标值落在该区间内

## 模型提示词设计

当前的提示词（system prompt）指导模型：
1. 以医学影像专家身份分析超声图像
2. 按0/1进行二分类
3. 输出结构化JSON格式
4. 包含置信度分数和推理过程

可通过修改`classify_with_gpt()`函数中的`system_prompt`和`user_prompt`来定制。

## 常见问题

### Q: 如何调整分类阈值？
```bash
python gpt_thyroid_binary_eval.py ... --threshold 0.7
```
默认为0.5。提高阈值会增加特异性，降低敏感性。

### Q: 如何降低API成本？
- 使用`gpt-4o-mini`模型
- 使用`--limit`参数做小规模测试
- 降低`--ci_bootstrap`值（如1000）以加快置信区间计算

### Q: 如何处理API速率限制？
脚本已内置指数退避重试机制。调整参数：
```bash
--max_retries 5 --retry_delay 3.0
```

### Q: 模型响应格式错误怎么办？
脚本包含JSON解析容错机制，会尝试从响应中提取JSON。如果完全失败，会返回中性预测（0.5）。

### Q: 为什么置信度分数总是整数？
某些模型可能返回离散的置信度值（0, 0.5, 1）。可通过更详细的提示词改进。

## 与开源模型的对比

项目中还包含其他开源模型的评估：

| 脚本 | 模型 | 类型 | 特点 |
|------|------|------|------|
| `llama_thyroid_inference.py` | Llama 3.2 Vision | 开源 | 本地运行，低成本 |
| `qwen3_vl_thyroid_binary_eval.py` | Qwen 3-VL | 开源 | 多模态，流畅 |
| `medgemma_thyroid_binary_eval.py` | MedGemma | 开源/医学微调 | 医学专业，小模型 |
| `gpt_thyroid_binary_eval.py` | GPT-4o | 闭源 | 最强性能，付费 |

## 高级用法

### 多模型对比评估

见`compare_models.py`脚本（如果存在），支持在同一数据集上评估多个模型。

### 自定义评估指标

修改`compute_metrics()`函数以添加额外指标：
```python
def compute_metrics(y_true, y_prob, y_pred):
    # ... existing metrics ...
    
    # 添加自定义指标
    specificity_at_90_sensitivity = ...  # 自定义计算
    
    return {
        # ... existing ...
        "specificity_at_90_sensitivity": value,
    }
```

### 阈值扫描

```python
thresholds = np.arange(0.3, 0.8, 0.05)
for threshold in thresholds:
    cis = bootstrap_metric_cis(..., threshold=threshold)
    # 保存结果用于阈值-性能曲线
```

## 参考资源

- [OpenAI API 文档](https://platform.openai.com/docs)
- [GPT-4 Vision 指南](https://platform.openai.com/docs/guides/vision)
- [Bootstrap置信区间](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

## 许可证

本脚本遵循项目许可证。使用OpenAI API需遵守其[使用条款](https://openai.com/policies/terms-of-use)。

---

**提示**：在使用GPT-4 API时，请注意成本。建议先用`gpt-4o-mini`做原型开发。
