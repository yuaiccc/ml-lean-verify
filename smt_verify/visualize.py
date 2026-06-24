#!/usr/bin/env python3
"""
Step 4: 可视化 SMT + PGD 验证结果

生成：
1. SMT vs PGD 对比热力图
2. 对抗样本可视化（原始 vs 对抗）
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── 网络定义 ──────────────────────────────────────────────

class TinyMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── 颜色定义 ─────────────────────────────────────────────

COLORS = {
    "safe": "#2ecc71",       # 绿色 — 已证明鲁棒
    "unsafe": "#e74c3c",     # 红色 — 找到对抗样本
    "unknown": "#f39c12",    # 橙色 — 无法判定
    "pgd_success": "#e74c3c",
    "pgd_fail": "#95a5a6",
}

# ── 热力图 ───────────────────────────────────────────────

def plot_heatmap(smt_results, pgd_results, save_path):
    """SMT vs PGD 对比热力图"""
    n_samples = 5
    epsilons = [0.02, 0.05, 0.1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (ax, results, title) in enumerate([
        (axes[0], smt_results, "Z3 SMT 验证"),
        (axes[1], pgd_results, "PGD 攻击"),
    ]):
        # 构建矩阵
        matrix = np.zeros((n_samples, len(epsilons)))
        labels = np.empty((n_samples, len(epsilons)), dtype=object)

        for r in results:
            si = r["sample_idx"]
            ei = epsilons.index(r["epsilon"])
            if "status" in r:
                if r["status"] == "safe":
                    matrix[si, ei] = 0
                    labels[si, ei] = "SAFE"
                elif r["status"] == "unsafe":
                    matrix[si, ei] = 1
                    labels[si, ei] = "UNSAFE"
                else:
                    matrix[si, ei] = 2
                    labels[si, ei] = "UNKNOWN"
            elif "pgd_success" in r:
                if r["pgd_success"]:
                    matrix[si, ei] = 1
                    labels[si, ei] = f"ADV→{r['adv_prediction']}"
                else:
                    matrix[si, ei] = 2
                    labels[si, ei] = "robust"

        # 自定义颜色映射
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([COLORS["safe"], COLORS["unsafe"], COLORS["unknown"]])

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

        # 添加标签
        for i in range(n_samples):
            for j in range(len(epsilons)):
                text_color = "white" if matrix[i, j] != 2 else "black"
                ax.text(j, i, labels[i, j], ha="center", va="center",
                       fontsize=9, fontweight='bold', color=text_color)

        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"ε={e}" for e in epsilons])
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels([f"Sample {i}" for i in range(n_samples)])
        ax.set_title(title, fontsize=14, fontweight='bold')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["safe"], label="SAFE (已证明鲁棒)"),
        Patch(facecolor=COLORS["unsafe"], label="UNSAFE (找到对抗样本)"),
        Patch(facecolor=COLORS["unknown"], label="UNKNOWN (无法判定)"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("MNIST TinyMLP (784→8→10) L∞ 鲁棒性验证", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  热力图已保存: {save_path}")
    plt.close()

# ── 对抗样本可视化 ────────────────────────────────────────

def plot_adversarial_examples(model, samples, pgd_results, save_path):
    """可视化对抗样本"""
    # 找到 PGD 成功的样本
    adv_cases = [r for r in pgd_results if r["pgd_success"]]
    n_adv = len(adv_cases)

    if n_adv == 0:
        print("  无对抗样本可可视化")
        return

    # 选择最多 6 个对抗样本
    adv_cases = adv_cases[:6]
    n_show = len(adv_cases)

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for idx, case in enumerate(adv_cases):
        sample = samples[case["sample_idx"]]
        x_orig = sample["image"].reshape(28, 28)
        label = case["label"]
        eps = case["epsilon"]
        adv_pred = case["adv_prediction"]

        # 重新生成对抗样本
        x_tensor = torch.tensor(sample["image"], dtype=torch.float32).view(1, 1, 28, 28)
        alpha = eps / 4.0
        x_adv = x_tensor.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        criterion = nn.CrossEntropyLoss()
        for step in range(200):
            x_adv.requires_grad_(True)
            output = model(x_adv)
            loss = criterion(output, torch.tensor([label]))
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.max(torch.min(x_adv, x_tensor + eps), x_tensor - eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
            if model(x_adv).argmax().item() != label:
                break

        x_adv_np = x_adv.view(28, 28).numpy()
        perturbation = np.abs(x_adv_np - x_orig)

        # 原始图像
        axes[0, idx].imshow(x_orig, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f"Original\nlabel={label}", fontsize=10)
        axes[0, idx].axis('off')

        # 对抗图像
        axes[1, idx].imshow(x_adv_np, cmap='gray', vmin=0, vmax=1)
        axes[1, idx].set_title(f"Adversarial\npred={adv_pred}, ε={eps}", fontsize=10, color='red')
        axes[1, idx].axis('off')

    fig.suptitle("PGD 对抗样本示例（人眼几乎看不出区别）", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  对抗样本可视化已保存: {save_path}")
    plt.close()

# ── 主函数 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 4: 可视化验证结果")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)

    # 加载结果
    smt_path = os.path.join(base_dir, "verification_results.json")
    pgd_path = os.path.join(base_dir, "pgd_results.json")

    with open(smt_path) as f:
        smt_results = json.load(f)
    with open(pgd_path) as f:
        pgd_results = json.load(f)

    # 加载模型和样本
    model = TinyMLP(hidden=8)
    model.load_state_dict(torch.load(os.path.join(base_dir, "mnist_tiny.pt"), weights_only=True))
    model.eval()

    samples = np.load(os.path.join(base_dir, "test_samples.npy"), allow_pickle=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 生成热力图
    heatmap_path = os.path.join(base_dir, "verification_heatmap.png")
    plot_heatmap(smt_results, pgd_results, heatmap_path)

    # 生成对抗样本可视化
    adv_path = os.path.join(base_dir, "adversarial_examples.png")
    plot_adversarial_examples(model, samples, pgd_results, adv_path)

    print("\n  可视化完成！")

if __name__ == "__main__":
    main()
