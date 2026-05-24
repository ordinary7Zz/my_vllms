#!/usr/bin/env python3
"""
快速入门：使用GPT模型评估甲状腺分类

这个脚本展示了如何快速开始使用GPT进行甲状腺恶性肿瘤分类。
"""

import os
import sys
import json
from pathlib import Path


def check_requirements():
    """检查依赖是否已安装。"""
    print("检查依赖...")
    required = {
        'openai': 'pip install openai',
        'numpy': 'pip install numpy',
        'sklearn': 'pip install scikit-learn',
        'PIL': 'pip install pillow',
        'tqdm': 'pip install tqdm',
    }
    
    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package if package != 'sklearn' else 'sklearn')
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(install_cmd)
    
    if missing:
        print("\n请安装缺失的包：")
        for cmd in missing:
            print(f"  {cmd}")
        return False
    return True


def check_api_key():
    """检查OpenAI API密钥。"""
    print("\n检查OpenAI API密钥...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("  ✗ OPENAI_API_KEY 环境变量未设置")
        print("\n  设置方法:")
        print("    Linux/Mac: export OPENAI_API_KEY='sk-xxxxx'")
        print("    Windows (PowerShell): $env:OPENAI_API_KEY='sk-xxxxx'")
        print("    Windows (CMD): set OPENAI_API_KEY=sk-xxxxx")
        return False
    
    if api_key.startswith('sk-'):
        print(f"  ✓ API密钥已设置 (前缀: sk-...)")
        return True
    else:
        print(f"  ✗ API密钥格式不正确 (应以 'sk-' 开头)")
        return False


def verify_data_structure():
    """验证数据结构。"""
    print("\n验证数据结构...")
    
    # 检查图像目录
    image_dirs = [
        './test_images',
        '../test_images',
        'D:/ThyroidDataset/test_images',  # 修改为你的实际路径
    ]
    
    found_images = False
    for img_dir in image_dirs:
        if os.path.isdir(img_dir):
            image_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✓ 找到图像目录: {img_dir} ({image_count} 张图片)")
            found_images = True
            break
    
    if not found_images:
        print(f"  ✗ 未找到图像目录")
        print(f"    请检查以下路径是否存在:")
        for path in image_dirs:
            print(f"      - {path}")
    
    # 检查标签文件
    label_files = [
        './test_labels.json',
        '../test_labels.json',
        'D:/ThyroidDataset/test_labels.json',  # 修改为你的实际路径
    ]
    
    found_labels = False
    for label_file in label_files:
        if os.path.isfile(label_file):
            try:
                with open(label_file) as f:
                    labels = json.load(f)
                    print(f"  ✓ 找到标签文件: {label_file} ({len(labels)} 个样本)")
                    found_labels = True
                    break
            except:
                print(f"  ✗ 标签文件格式错误: {label_file}")
    
    if not found_labels:
        print(f"  ✗ 未找到标签文件")
        print(f"    标签文件应为JSON格式，包含 filename 和 malignancy 字段")
    
    return found_images and found_labels


def show_quick_start():
    """显示快速开始指南。"""
    print("\n" + "=" * 80)
    print("快速开始指南")
    print("=" * 80)
    
    print("""
1. 基本用法（需要修改路径）：

   python gpt/gpt_thyroid_binary_eval.py \\
       --image_dir /path/to/images \\
       --label_json /path/to/labels.json \\
       --out_csv results.csv


2. 测试运行（仅处理5个样本）：

   python gpt/gpt_thyroid_binary_eval.py \\
       --image_dir /path/to/images \\
       --label_json /path/to/labels.json \\
       --out_csv test_results.csv \\
       --limit 5


3. 使用特定模型：

   # 使用成本较低的模型
   python gpt/gpt_thyroid_binary_eval.py \\
       --image_dir /path/to/images \\
       --label_json /path/to/labels.json \\
       --model gpt-4o-mini \\
       --out_csv results.csv

   # 使用最强的模型
   python gpt/gpt_thyroid_binary_eval.py \\
       --image_dir /path/to/images \\
       --label_json /path/to/labels.json \\
       --model gpt-4-vision-preview \\
       --out_csv results.csv


4. 调整分类阈值：

   python gpt/gpt_thyroid_binary_eval.py \\
       --image_dir /path/to/images \\
       --label_json /path/to/labels.json \\
       --threshold 0.7 \\
       --out_csv results.csv


5. 多模型对比（已有多个模型的预测结果）：

   python compare_models.py


6. 调整Bootstrap置信区间参数：

   python gpt/gpt_thyroid_binary_eval.py \\
       --image_dir /path/to/images \\
       --label_json /path/to/labels.json \\
       --ci_bootstrap 5000 \\
       --ci_alpha 0.99 \\
       --out_csv results.csv
    """)


def show_example_data_structure():
    """显示示例数据结构。"""
    print("\n" + "=" * 80)
    print("数据结构示例")
    print("=" * 80)
    
    print("\n目录结构:")
    print("""
my_vllms/
├── gpt/
│   ├── gpt_thyroid_binary_eval.py       # GPT评估脚本
│   ├── gpt_thyroid_eval.sh              # Bash脚本
│   └── README_GPT.md                    # 详细文档
├── llama-3.2/                           # LLaMA模型
├── qwen3/                               # Qwen3模型
├── medgemma/                            # MedGemma模型
├── compare_models.py                    # 多模型对比
└── test_data/
    ├── images/                          # 图像文件
    │   ├── patient001.jpg
    │   ├── patient002.jpg
    │   └── ...
    └── labels.json                      # 标签文件
    """)
    
    print("\n标签JSON格式示例:")
    example_labels = [
        {"filename": "patient001.jpg", "malignancy": 0},
        {"filename": "patient002.jpg", "malignancy": 1},
        {"filename": "patient003.jpg", "malignancy": 0},
    ]
    print(json.dumps(example_labels, indent=2, ensure_ascii=False))
    
    print("\n输出CSV示例:")
    print("""
filename,gt_malignant,p_malignant,pred_malignant,parsed_prediction,reasoning
patient001.jpg,0,0.143,0,0,Benign characteristics observed
patient002.jpg,1,0.857,1,1,Suspicious features detected
patient003.jpg,0,0.286,0,0,Clear benign pattern
    """)


def main():
    """主函数。"""
    print("\n" + "=" * 80)
    print("GPT甲状腺分类 - 快速入门检查")
    print("=" * 80)
    
    all_good = True
    
    # 检查依赖
    if not check_requirements():
        all_good = False
    
    # 检查API密钥
    if not check_api_key():
        all_good = False
    
    # 验证数据
    if not verify_data_structure():
        all_good = False
    
    # 显示快速开始
    show_quick_start()
    show_example_data_structure()
    
    print("\n" + "=" * 80)
    if all_good:
        print("✓ 所有检查通过！您可以开始运行GPT评估脚本。")
    else:
        print("✗ 存在未通过的检查项。请按上述提示进行配置。")
    print("=" * 80 + "\n")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
