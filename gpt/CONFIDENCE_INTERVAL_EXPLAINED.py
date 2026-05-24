"""
闭源模型置信区间计算原理详解

说明：置信区间的计算与模型类型（闭源/开源）无关，而是基于统计学的Bootstrap重采样方法。
"""

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


# ============================================================================
# 1. Bootstrap置信区间的基本原理
# ============================================================================

"""
Bootstrap置信区间计算步骤：

1. 从原始数据集（n个样本）中有放回地随机抽取n个样本 → 形成Bootstrap样本
2. 用该样本计算感兴趣的指标（如AUROC、F1等）
3. 重复步骤1-2共B次（通常B=2000），得到B个指标值
4. 计算这B个指标值的分位数，得到置信区间

对于95%置信区间（alpha=0.95）：
- 下界 = 第2.5分位数（对应 (1-0.95)/2 = 0.025）
- 上界 = 第97.5分位数（对应 1 - 0.025 = 0.975）

数学表示：
  CI = [Q_0.025(指标分布), Q_0.975(指标分布)]
  其中Q_p表示第p分位数
"""

# ============================================================================
# 2. 代码实现详解
# ============================================================================

def bootstrap_ci_explained(
    y_true: List[int],
    y_prob: List[float],
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap置信区间计算函数 - 详细版本
    
    核心思想：
    =========
    如果我们的数据集是从某个总体中随机抽样得到的，
    那么通过Bootstrap重采样可以估计该总体统计量的分布，
    进而计算其置信区间。
    
    参数解释：
    =========
    y_true: 真实标签 [0, 1, 0, 1, ...] （长度n）
    y_prob: 模型预测概率 [0.1, 0.9, 0.2, 0.8, ...] （长度n）
    threshold: 分类阈值（默认0.5），pred = (prob >= threshold)
    n_bootstrap: Bootstrap样本数（默认2000，越多越精确但越慢）
    alpha: 置信水平（0.95表示95% CI）
    seed: 随机种子（保证可重复性）
    
    返回值：
    =======
    {
        'auroc': (下界, 上界),
        'accuracy': (下界, 上界),
        ...
    }
    """
    
    # 第一步：准备数据
    # ───────────────
    y_true_arr = np.asarray(y_true)     # 转为numpy数组
    y_prob_arr = np.asarray(y_prob)
    n = len(y_true_arr)                 # 样本总数
    
    print(f"原始数据集大小: {n} 个样本")
    print(f"Bootstrap重采样次数: {n_bootstrap}")
    print(f"置信水平: {alpha*100:.0f}%")
    
    # 初始化结果字典
    metric_samples = {
        "auroc": [],      # 存储2000次Bootstrap的AUROC值
        "accuracy": [],   # 存储2000次Bootstrap的准确率值
        "f1": [],         # ...
        "sensitivity": [],
        "specificity": [],
    }
    
    # 第二步：Bootstrap重采样循环
    # ───────────────────────────
    rng = np.random.default_rng(seed)   # 创建随机数生成器
    
    for b in range(n_bootstrap):
        # 步骤2.1: 有放回地随机抽取n个样本的索引
        # 关键：可能重复抽取同一样本，也可能漏掉某些样本
        indices = rng.integers(0, n, size=n)
        # 例如：indices可能是 [5, 2, 5, 0, 3, 1, ...] （长度=n）
        
        # 步骤2.2: 根据索引抽取样本
        sample_y_true = y_true_arr[indices]    # 长度为n的子样本
        sample_y_prob = y_prob_arr[indices]
        
        # 步骤2.3: 用子样本的预测概率生成预测标签
        sample_y_pred = (sample_y_prob >= threshold).astype(int)
        
        # 步骤2.4: 计算该子样本上的所有指标
        # （这是一个虚拟函数，真实实现见下面）
        metrics = compute_metrics_example(
            sample_y_true.tolist(),
            sample_y_prob.tolist(),
            sample_y_pred.tolist(),
        )
        
        # 步骤2.5: 保存各指标值
        for key in metric_samples:
            value = metrics[key]
            if not np.isnan(value):  # 只保存有效的值（非NaN）
                metric_samples[key].append(value)
        
        if (b + 1) % 500 == 0:
            print(f"  已完成 {b+1}/{n_bootstrap} 次重采样")
    
    # 第三步：计算分位数
    # ──────────────────
    lower_q = (1.0 - alpha) / 2.0    # 对于95% CI: 0.025
    upper_q = 1.0 - lower_q           # 对于95% CI: 0.975
    
    print(f"\n计算第{lower_q*100:.1f}%和{upper_q*100:.1f}%分位数...")
    
    cis: Dict[str, Tuple[float, float]] = {}
    for key, values in metric_samples.items():
        if len(values) > 0:
            # 使用np.quantile计算分位数
            lower = float(np.quantile(values, lower_q))
            upper = float(np.quantile(values, upper_q))
            cis[key] = (lower, upper)
            
            mean_value = float(np.mean(values))
            print(f"{key:12s}: 均值={mean_value:.4f}, "
                  f"95% CI=[{lower:.4f}, {upper:.4f}]")
        else:
            cis[key] = (float("nan"), float("nan"))
    
    return cis


def compute_metrics_example(y_true, y_prob, y_pred):
    """计算指标的虚拟函数（实际使用sklearn）"""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, confusion_matrix
    )
    
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    
    return {
        'auroc': auroc,
        'accuracy': acc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


# ============================================================================
# 3. 为什么Bootstrap方法对闭源模型有效？
# ============================================================================

"""
关键原因：
========

1. 只需要输入和输出，不需要模型内部结构
   ├─ 输入：真实标签 y_true
   ├─ 输入：预测概率 y_prob
   └─ 不需要：模型权重、梯度、架构等

2. Bootstrap是一种非参数统计方法
   ├─ 无需假设数据分布（正态、伯努利等）
   ├─ 对任何模型的输出都适用
   └─ 包括：LLM、神经网络、传统ML等

3. 对样本量的要求较低
   ├─ n ≥ 30通常就足够好
   ├─ n ≥ 100时效果很好
   └─ n < 30时仍优于参数方法

示意图：
======

闭源模型（GPT-4o）        |    开源模型（LLaMA）
─────────────────────────┼──────────────────────
输入图像 → [API调用]      |    输入图像 → [本地推理]
        → [JSON响应]      |              → [logits]
        → 提取概率p_pred  |              → 提取概率p_pred
                 ↓        |                      ↓
          y_prob = [0.1, 0.9, 0.2, ...]  (都是相同格式)
                 ↓        |                      ↓
          Bootstrap重采样 |         Bootstrap重采样
          计算置信区间     |         计算置信区间
          ← 结果相同 →    |
"""

# ============================================================================
# 4. 数值例子
# ============================================================================

def example_bootstrap_visualization():
    """
    实际例子：10个样本的Bootstrap置信区间计算
    """
    print("\n" + "="*80)
    print("数值例子：10个样本的Bootstrap置信区间")
    print("="*80)
    
    # 真实数据
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.85, 0.15, 0.92, 0.05])
    
    print(f"\n原始数据 (n=10):")
    print(f"y_true = {y_true}")
    print(f"y_prob = {y_prob.round(2)}")
    
    # 手工演示几次Bootstrap重采样
    np.random.seed(42)
    print(f"\nBootstrap演示 (前5次重采样):")
    print("-" * 80)
    
    accuracies = []
    for b in range(5):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        sample_y_true = y_true[indices]
        sample_y_prob = y_prob[indices]
        sample_y_pred = (sample_y_prob >= 0.5).astype(int)
        
        acc = (sample_y_true == sample_y_pred).sum() / len(sample_y_true)
        accuracies.append(acc)
        
        print(f"第{b+1}次: 索引={indices} → 准确率={acc:.2f}")
    
    # 完整Bootstrap
    print(f"\n执行2000次Bootstrap重采样...")
    accuracies_full = []
    for b in range(2000):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        sample_y_true = y_true[indices]
        sample_y_prob = y_prob[indices]
        sample_y_pred = (sample_y_prob >= 0.5).astype(int)
        acc = (sample_y_true == sample_y_pred).sum() / len(sample_y_true)
        accuracies_full.append(acc)
    
    # 计算置信区间
    ci_lower = np.quantile(accuracies_full, 0.025)
    ci_upper = np.quantile(accuracies_full, 0.975)
    ci_mean = np.mean(accuracies_full)
    
    print(f"\n结果:")
    print(f"  准确率均值: {ci_mean:.4f}")
    print(f"  95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  区间宽度: {ci_upper - ci_lower:.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 子图1：Bootstrap值的分布
    plt.subplot(1, 2, 1)
    plt.hist(accuracies_full, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(ci_mean, color='red', linestyle='--', linewidth=2, label=f'均值={ci_mean:.3f}')
    plt.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label=f'下界={ci_lower:.3f}')
    plt.axvline(ci_upper, color='green', linestyle='--', linewidth=2, label=f'上界={ci_upper:.3f}')
    plt.xlabel('准确率')
    plt.ylabel('频数')
    plt.title('2000次Bootstrap准确率的分布')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图2：累积分布
    plt.subplot(1, 2, 2)
    sorted_accs = np.sort(accuracies_full)
    plt.plot(range(len(sorted_accs)), sorted_accs, linewidth=2)
    plt.axhline(ci_lower, color='green', linestyle='--', label='95% CI边界')
    plt.axhline(ci_upper, color='green', linestyle='--')
    plt.xlabel('排序样本编号')
    plt.ylabel('准确率')
    plt.title('Bootstrap值的累积分布')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bootstrap_example.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: bootstrap_example.png")
    plt.close()


# ============================================================================
# 5. 不同置信水平的比较
# ============================================================================

def confidence_levels_comparison():
    """不同置信水平的比较"""
    print("\n" + "="*80)
    print("不同置信水平的比较")
    print("="*80)
    
    # 生成模拟数据
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, size=100)
    y_prob = np.random.uniform(0, 1, size=100)
    
    # 计算不同置信水平的CI
    accuracies = []
    for _ in range(2000):
        indices = np.random.choice(100, 100, replace=True)
        y_pred = (y_prob[indices] >= 0.5).astype(int)
        acc = (y_true[indices] == y_pred).mean()
        accuracies.append(acc)
    
    print(f"\n准确率的Bootstrap分布 (2000次重采样):")
    print(f"  均值: {np.mean(accuracies):.4f}")
    print(f"  标准差: {np.std(accuracies):.4f}")
    
    alphas = [0.90, 0.95, 0.99]
    for alpha in alphas:
        lower_q = (1 - alpha) / 2
        upper_q = 1 - lower_q
        lower = np.quantile(accuracies, lower_q)
        upper = np.quantile(accuracies, upper_q)
        width = upper - lower
        print(f"  {alpha*100:.0f}% CI: [{lower:.4f}, {upper:.4f}] (宽度={width:.4f})")
    
    print(f"\n观察：置信水平越高，置信区间越宽")
    print(f"  90% CI < 95% CI < 99% CI")


# ============================================================================
# 6. 与参数方法的对比
# ============================================================================

"""
Bootstrap vs 参数方法：
====================

Bootstrap方法                              参数方法（如正态分布）
─────────────────────────────────────────────────────────────
无需假设分布                               假设指标服从正态分布
对非对称分布有效                           对对称分布效果最好
需要更多计算（多次重采样）                 计算快速（解析公式）
不需要模型细节                             某些情况需要梯度信息
适合任何样本量                             小样本时不可靠
对异常值鲁棒性较强                         对异常值敏感


对于医学诊断应用（甲状腺分类）：
 → Bootstrap更合适，因为：
   1. 样本不一定正态分布
   2. 需要发表论文时的稳健性
   3. 不关心计算速度（只运行一次）
"""


# ============================================================================
# 7. 如何选择Bootstrap参数
# ============================================================================

"""
n_bootstrap 参数选择：
====================

n_bootstrap=1000    └─ 快速原型，精度足够（标准误≈0.001-0.01）
n_bootstrap=2000    └─ 推荐默认值，平衡精度和速度（标准误≈0.0005）
n_bootstrap=5000    └─ 论文发表级别，更高精度（标准误≈0.0002）
n_bootstrap=10000   └─ 极高精度，但很慢（标准误≈0.0001）

经验法则：
  如果有N个样本，通常 B ≥ N 就足够了
  对于100个样本的数据，B=2000已经是过度的

alpha 参数选择：
==============

alpha=0.90    └─ 90% 置信区间，较窄，取舍空间大
alpha=0.95    └─ 95% 置信区间，最常用（医学标准）
alpha=0.99    └─ 99% 置信区间，较宽，更保守
"""


if __name__ == "__main__":
    # 数值例子
    example_bootstrap_visualization()
    
    # 置信水平对比
    confidence_levels_comparison()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
1. 置信区间计算方法与模型是否闭源无关
   └─ 只需要：真实标签 + 预测概率
   
2. Bootstrap是非参数方法，适合所有情况
   └─ 无分布假设，对小样本稳健
   
3. 默认参数（n_bootstrap=2000, alpha=0.95）经过充分验证
   └─ 用于学术论文和临床应用
   
4. 计算置信区间的时间复杂度为O(B×n)
   └─ B=2000, n=100 → 20万次指标计算（<1秒）
    """)
