#!/usr/bin/env python3
"""
Step 3: PGD 对抗攻击 — 寻找对抗样本

与 Z3 SMT 验证互补：
- Z3 (形式化验证)：证明鲁棒性 (SAFE) 或无法判定 (UNKNOWN) — 数学保证
- PGD (梯度攻击)：寻找对抗样本 — 实际攻击，无形式化保证

如果 PGD 找到对抗样本 → 网络确定不鲁棒（UNSAFE）
如果 PGD 找不到 → 不代表鲁棒（可能攻击不够强）
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os

# ── 网络定义 ──────────────────────────────────────────────

class TinyMLP(nn.Module):
    """784 → 8 (ReLU) → 10"""
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── PGD 攻击 ─────────────────────────────────────────────

def pgd_attack(model, x_orig, label, epsilon, alpha=None, n_steps=100):
    """
    PGD (Projected Gradient Descent) L∞ 对抗攻击。

    参数：
        model: 目标模型
        x_orig: 原始输入 [1, 1, 28, 28]
        label: 正确标签
        epsilon: L∞ 扰动半径
        alpha: 步长（默认 epsilon/4）
        n_steps: 迭代步数

    返回：
        (success, x_adv, adv_pred, n_steps_used)
    """
    if alpha is None:
        alpha = epsilon / 4.0

    x_adv = x_orig.clone().detach()
    # 随机起点
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    criterion = nn.CrossEntropyLoss()

    for step in range(n_steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = criterion(output, torch.tensor([label]))
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # 梯度上升（最大化 loss = 让分类错误）
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # 投影到 L∞ 球内
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            # 裁剪到合法范围
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        # 检查是否已成功
        pred = model(x_adv).argmax().item()
        if pred != label:
            return (True, x_adv.view(-1).numpy(), pred, step + 1)

    return (False, None, None, n_steps)

# ── 批量攻击 ─────────────────────────────────────────────

def run_pgd_attacks(model, samples, epsilons):
    results = []

    for sample_idx, sample in enumerate(samples):
        x_orig = sample["image"]
        label = sample["label"]
        x_tensor = torch.tensor(x_orig, dtype=torch.float32).view(1, 1, 28, 28)

        for eps in epsilons:
            success, x_adv, adv_pred, n_steps = pgd_attack(
                model, x_tensor, label, eps, n_steps=200
            )

            if success:
                l_inf = float(np.max(np.abs(x_adv - x_orig)))
                print(f"  Sample {sample_idx} (label={label}), ε={eps:.4f} → "
                      f"ADVERSARIAL (pred={adv_pred}, L∞={l_inf:.4f}, {n_steps} steps)")
            else:
                print(f"  Sample {sample_idx} (label={label}), ε={eps:.4f} → "
                      f"robust (200 steps failed)")

            results.append({
                "sample_idx": int(sample_idx),
                "label": int(label),
                "epsilon": float(eps),
                "pgd_success": bool(success),
                "adv_prediction": int(adv_pred) if success else None,
                "adv_distance": float(np.max(np.abs(x_adv - x_orig))) if success else None,
                "n_steps": int(n_steps),
            })

    return results

# ── 主函数 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 3: PGD 对抗攻击 (互补 SMT 验证)")
    print("=" * 60)

    model = TinyMLP(hidden=8)
    model_path = os.path.join(os.path.dirname(__file__), "mnist_tiny.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    samples_path = os.path.join(os.path.dirname(__file__), "test_samples.npy")
    samples = np.load(samples_path, allow_pickle=True)

    n_samples = min(5, len(samples))
    samples = samples[:n_samples]
    epsilons = [0.02, 0.05, 0.1]

    print(f"\n  Model: TinyMLP (784→8→10)")
    print(f"  Attack: PGD L∞ (200 steps, α=ε/4)")
    print(f"  Samples: {n_samples}")
    print(f"  Epsilons: {epsilons}")
    print()

    results = run_pgd_attacks(model, samples, epsilons)

    # 保存结果
    results_path = os.path.join(os.path.dirname(__file__), "pgd_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总
    print("\n" + "=" * 60)
    print("PGD 攻击结果汇总")
    print("=" * 60)

    success_count = sum(1 for r in results if r["pgd_success"])
    fail_count = sum(1 for r in results if not r["pgd_success"])

    print(f"  攻击成功: {success_count}/{len(results)}")
    print(f"  攻击失败: {fail_count}/{len(results)}")

    print(f"\n  {'Sample':>8s}  {'Label':>6s}  {'ε':>8s}  {'PGD':>10s}  {'Adv Pred':>10s}  {'L∞':>8s}  {'Steps':>6s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")
    for r in results:
        pgd_str = "SUCCESS" if r["pgd_success"] else "failed"
        adv_str = str(r["adv_prediction"]) if r["adv_prediction"] is not None else "-"
        dist_str = f"{r['adv_distance']:.4f}" if r["adv_distance"] is not None else "-"
        print(f"  {r['sample_idx']:8d}  {r['label']:6d}  {r['epsilon']:8.4f}  {pgd_str:>10s}  {adv_str:>10s}  {dist_str:>8s}  {r['n_steps']:6d}")

    print(f"\n  Results saved to {results_path}")

    # 对比 SMT 结果
    smt_path = os.path.join(os.path.dirname(__file__), "verification_results.json")
    if os.path.exists(smt_path):
        with open(smt_path) as f:
            smt_results = json.load(f)

        print("\n" + "=" * 60)
        print("SMT vs PGD 对比")
        print("=" * 60)
        print(f"  {'Sample':>8s}  {'Label':>6s}  {'ε':>8s}  {'SMT':>10s}  {'PGD':>10s}  {'结论':>20s}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*20}")

        for smt_r in smt_results:
            matching_pgd = next(
                (p for p in results
                 if p["sample_idx"] == smt_r["sample_idx"] and abs(p["epsilon"] - smt_r["epsilon"]) < 1e-6),
                None
            )
            if matching_pgd:
                smt_str = smt_r["status"]
                pgd_str = "SUCCESS" if matching_pgd["pgd_success"] else "failed"

                if smt_r["status"] == "safe" and not matching_pgd["pgd_success"]:
                    conclusion = "✓ 鲁棒（已证明）"
                elif smt_r["status"] == "safe" and matching_pgd["pgd_success"]:
                    conclusion = "✗ 矛盾（需检查）"
                elif smt_r["status"] == "unknown" and matching_pgd["pgd_success"]:
                    conclusion = "✗ 不鲁棒（PGD反例）"
                elif smt_r["status"] == "unknown" and not matching_pgd["pgd_success"]:
                    conclusion = "? 无法判定"
                else:
                    conclusion = "-"

                print(f"  {smt_r['sample_idx']:8d}  {smt_r['label']:6d}  {smt_r['epsilon']:8.4f}  {smt_str:>10s}  {pgd_str:>10s}  {conclusion:>20s}")

if __name__ == "__main__":
    main()
