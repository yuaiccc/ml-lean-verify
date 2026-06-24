#!/usr/bin/env python3
"""
Step 5: 用 Z3 SMT solver 重新验证 IBP 训练的鲁棒模型

复用 verify_robustness.py 的 Z3 编码逻辑，但：
  - 加载 mnist_robust.pt（IBP 训练）而非 mnist_tiny.pt（标准训练）
  - 加载 test_samples_robust.npy 而非 test_samples.npy
  - 增量保存结果到 verification_results_robust.json
  - 打印对比表格：原始模型 vs IBP 训练模型的 SAFE 数量

验证属性：对于输入 x（正确分类为 label c），
在 L∞ 扰动 ε 下，网络输出 y_correct > y_other（对所有 other ≠ c）。

Z3 返回 unsat → SAFE（鲁棒，数学证明）
Z3 返回 sat   → UNSAFE（找到对抗样本）
Z3 返回 unknown → 无法判定
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
import os

from z3 import (
    Real, RealVal, RealVector, Solver, And, Or, Not, If, sat, unsat, unknown,
    simplify, Model
)

# ── 网络定义（与其它脚本一致）──────────────────────────────

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

# ── 提取权重 ──────────────────────────────────────────────

def extract_weights(model):
    weights = {}
    biases = {}
    with torch.no_grad():
        weights['fc1'] = model.fc1.weight.numpy()  # [8, 784]
        biases['fc1'] = model.fc1.bias.numpy()      # [8]
        weights['fc2'] = model.fc2.weight.numpy()  # [10, 8]
        biases['fc2'] = model.fc2.bias.numpy()      # [10]
    return weights, biases

# ── 活跃像素检测 ──────────────────────────────────────────

def get_active_pixels(x_orig, dilation=1):
    """检测活跃像素：原始值 > 阈值，加上膨胀邻居。"""
    active_mask = np.abs(x_orig) > 0.01
    if dilation > 0:
        try:
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(active_mask.reshape(28, 28), iterations=dilation)
            active_mask = dilated.reshape(-1)
        except Exception:
            pass
    active_indices = np.where(active_mask)[0]
    return active_indices

# ── Z3 编码（稀疏版：只为活跃像素创建变量）──────────────────

def encode_linear_layer_sparse(z3_input, input_indices, weight, bias, output_name, n_out):
    """
    编码线性层，只为活跃像素创建 Z3 变量。
    非活跃像素的贡献（weight * 0 = 0）被跳过。
    """
    n_active = len(z3_input)
    z3_output = RealVector(f"{output_name}", n_out)
    constraints = []
    for j in range(n_out):
        expr = RealVal(float(bias[j]))
        for k in range(n_active):
            i = input_indices[k]
            w = float(weight[j][i])
            if abs(w) > 1e-8:
                expr = expr + z3_input[k] * RealVal(w)
        constraints.append(z3_output[j] == expr)
    return z3_output, constraints


def encode_relu(z3_var, z3_output):
    """编码 ReLU：output = max(0, input)。"""
    return [z3_output[i] == If(z3_var[i] >= 0, z3_var[i], RealVal(0))
            for i in range(len(z3_var))]


def build_network_constraints(weights, biases, x_orig, epsilon, active_indices):
    """
    构建完整网络约束（稀疏版）：
    输入约束（仅活跃像素）→ fc1 → ReLU → fc2 → logits
    """
    constraints = []
    n_active = len(active_indices)

    # 只为活跃像素创建 Z3 变量
    input_vars = RealVector("x", n_active)

    # 输入约束：活跃像素在 [max(0, x-ε), min(1, x+ε)] 范围内
    for k in range(n_active):
        i = active_indices[k]
        lo = max(0.0, float(x_orig[i]) - epsilon)
        hi = min(1.0, float(x_orig[i]) + epsilon)
        constraints.append(input_vars[k] >= RealVal(lo))
        constraints.append(input_vars[k] <= RealVal(hi))

    # Layer 1: fc1 (784 → 8) — 稀疏编码
    h1, c1 = encode_linear_layer_sparse(
        input_vars, active_indices, weights['fc1'], biases['fc1'], "h1", 8
    )
    constraints.extend(c1)

    # ReLU 1
    r1 = RealVector("r1", 8)
    constraints.extend(encode_relu(h1, r1))

    # Layer 2: fc2 (8 → 10) — 全连接，变量少
    logits, c2 = encode_linear_layer_sparse(
        r1, list(range(8)), weights['fc2'], biases['fc2'], "logit", 10
    )
    constraints.extend(c2)

    return logits, constraints, input_vars, n_active

# ── 鲁棒性验证 ────────────────────────────────────────────

def verify_robustness(model, x_orig, label, epsilon, timeout_ms=120000):
    """
    验证 L∞ 鲁棒性：在 ε 扰动下，分类是否不变。

    返回: (status, x_adv, elapsed, n_active)
      status: "safe" | "unsafe" | "unknown" | "timeout"
    """
    weights, biases = extract_weights(model)

    # 检测活跃像素
    active_indices = get_active_pixels(x_orig, dilation=1)
    n_active = len(active_indices)

    # 构建网络约束
    logits, net_constraints, input_vars, n_active = build_network_constraints(
        weights, biases, x_orig, epsilon, active_indices
    )

    # 属性否定：存在 other ≠ label 使得 logit[other] >= logit[label]
    negation = Or([logits[other] >= logits[label] for other in range(10) if other != label])

    solver = Solver()
    solver.set("timeout", timeout_ms)

    for c in net_constraints:
        solver.add(c)
    solver.add(negation)

    start_time = time.time()
    result = solver.check()
    elapsed = time.time() - start_time

    if result == unsat:
        return ("safe", None, elapsed, n_active)
    elif result == sat:
        m = solver.model()
        x_adv = np.zeros(784)
        for k in range(n_active):
            i = active_indices[k]
            val = m.evaluate(input_vars[k])
            try:
                x_adv[i] = float(val.as_fraction())
            except Exception:
                try:
                    x_adv[i] = float(str(val))
                except Exception:
                    x_adv[i] = 0.0
        return ("unsafe", x_adv, elapsed, n_active)
    elif result == unknown:
        return ("unknown", None, elapsed, n_active)
    else:
        return ("timeout", None, elapsed, n_active)

# ── 增量保存 ──────────────────────────────────────────────

def save_results(results, path):
    """增量保存结果到 JSON。"""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ── 批量验证 ─────────────────────────────────────────────

def run_verification(model, samples, epsilons, results_path):
    results = []

    # 尝试加载已有结果（支持断点续跑）
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                results = json.load(f)
            print(f"  已加载 {len(results)} 条已有结果，继续验证...\n")
        except Exception:
            results = []

    for sample_idx, sample in enumerate(samples):
        x_orig = sample["image"]
        label = sample["label"]

        for eps in epsilons:
            # 检查是否已有此结果
            already_done = any(
                r["sample_idx"] == sample_idx and abs(r["epsilon"] - eps) < 1e-6
                for r in results
            )
            if already_done:
                existing = next(
                    r for r in results
                    if r["sample_idx"] == sample_idx and abs(r["epsilon"] - eps) < 1e-6
                )
                print(f"  Sample {sample_idx} (label={label}), ε={eps:.4f} → {existing['status']} (cached)")
                continue

            # 分层超时：小 ε 更可能 SAFE，给更多时间
            if eps <= 0.02:
                timeout_ms = 180000  # 3 min
            elif eps <= 0.05:
                timeout_ms = 120000  # 2 min
            else:
                timeout_ms = 60000   # 1 min

            print(f"  Sample {sample_idx} (label={label}), ε={eps:.4f} ... ", end="", flush=True)

            status, x_adv, elapsed, n_active = verify_robustness(
                model, x_orig, label, eps, timeout_ms=timeout_ms
            )

            adv_pred = None
            adv_distance = None
            if status == "unsafe" and x_adv is not None:
                x_adv_tensor = torch.tensor(x_adv, dtype=torch.float32).view(1, 1, 28, 28)
                with torch.no_grad():
                    adv_pred = model(x_adv_tensor).argmax().item()
                adv_distance = float(np.max(np.abs(x_adv - x_orig)))

            if status == "safe":
                print(f"SAFE ({elapsed:.1f}s, {n_active} active pixels)")
            elif status == "unsafe":
                print(f"UNSAFE ({elapsed:.1f}s) → adv_pred={adv_pred}, L∞={adv_distance:.4f}")
            elif status == "unknown":
                print(f"UNKNOWN ({elapsed:.1f}s, {n_active} active pixels)")
            else:
                print(f"TIMEOUT ({elapsed:.1f}s)")

            result_entry = {
                "sample_idx": int(sample_idx),
                "label": int(label),
                "epsilon": float(eps),
                "status": status,
                "elapsed": float(elapsed),
                "n_active_pixels": int(n_active),
                "adv_prediction": int(adv_pred) if adv_pred is not None else None,
                "adv_distance": float(adv_distance) if adv_distance is not None else None,
                "timeout_ms": timeout_ms,
            }
            results.append(result_entry)

            # 增量保存
            save_results(results, results_path)

    return results

# ── 对比表格 ─────────────────────────────────────────────

def print_comparison(robust_results, orig_results_path):
    """打印原始模型 vs IBP 训练模型的对比表格。"""
    orig_results = []
    if os.path.exists(orig_results_path):
        try:
            with open(orig_results_path) as f:
                orig_results = json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("对比表格：原始模型 (标准训练) vs IBP 训练模型")
    print("=" * 70)

    # 逐条对比
    header = (f"  {'Sample':>7s} {'Label':>6s} {'ε':>7s} | "
              f"{'原始':>10s} {'IBP':>10s} | {'改善':>8s}")
    print(header)
    print(f"  {'─'*7} {'─'*6} {'─'*7} | {'─'*10} {'─'*10} | {'─'*8}")

    orig_safe = 0
    robust_safe = 0
    improved = 0

    for r_r in robust_results:
        # 找到原始模型对应结果
        match = next(
            (o for o in orig_results
             if o["sample_idx"] == r_r["sample_idx"] and abs(o["epsilon"] - r_r["epsilon"]) < 1e-6),
            None
        )
        orig_status = match["status"] if match else "N/A"
        robust_status = r_r["status"]

        if orig_status == "safe":
            orig_safe += 1
        if robust_status == "safe":
            robust_safe += 1

        # 改善判定：原始非 safe → IBP safe
        change = ""
        if orig_status != "safe" and robust_status == "safe":
            change = "+ SAFE"
            improved += 1
        elif orig_status == "safe" and robust_status != "safe":
            change = "- 退化"
        elif orig_status == "safe" and robust_status == "safe":
            change = "= 保持"

        print(f"  {r_r['sample_idx']:7d} {r_r['label']:6d} {r_r['epsilon']:7.4f} | "
              f"{orig_status:>10s} {robust_status:>10s} | {change:>8s}")

    print(f"  {'─'*7} {'─'*6} {'─'*7} | {'─'*10} {'─'*10} | {'─'*8}")
    print(f"  {'SAFE 总计':>21s} | {orig_safe:>10d} {robust_safe:>10d} | "
          f"{'+' + str(improved):>8s}")

    print(f"\n  原始模型 SAFE: {orig_safe}/{len(robust_results)}")
    print(f"  IBP 模型 SAFE: {robust_safe}/{len(robust_results)}")
    print(f"  新增 SAFE (unknown/unsafe → safe): {improved}")

# ── 主函数 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 5: Z3 SMT 验证 IBP 鲁棒模型 (784→8→10)")
    print("=" * 60)

    model = TinyMLP(hidden=8)
    model_path = os.path.join(os.path.dirname(__file__), "mnist_robust.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    samples_path = os.path.join(os.path.dirname(__file__), "test_samples_robust.npy")
    samples = np.load(samples_path, allow_pickle=True)

    n_samples = min(5, len(samples))
    samples = samples[:n_samples]

    epsilons = [0.02, 0.05, 0.1]

    print(f"\n  Model: TinyMLP (784→8→10) — IBP 训练")
    print(f"  Samples: {n_samples}")
    print(f"  Epsilons: {epsilons}")
    print(f"  Total queries: {n_samples * len(epsilons)}")
    print(f"  Optimization: sparse variable encoding (active pixels only)")
    print()

    results_path = os.path.join(os.path.dirname(__file__), "verification_results_robust.json")

    results = run_verification(model, samples, epsilons, results_path)

    # 汇总
    print("\n" + "=" * 60)
    print("IBP 模型验证结果汇总")
    print("=" * 60)

    safe_count = sum(1 for r in results if r["status"] == "safe")
    unsafe_count = sum(1 for r in results if r["status"] == "unsafe")
    unknown_count = sum(1 for r in results if r["status"] == "unknown")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")

    print(f"  SAFE:     {safe_count}/{len(results)}")
    print(f"  UNSAFE:   {unsafe_count}/{len(results)}")
    print(f"  UNKNOWN:  {unknown_count}/{len(results)}")
    print(f"  TIMEOUT:  {timeout_count}/{len(results)}")

    print(f"\n  {'Sample':>8s}  {'Label':>6s}  {'ε':>8s}  {'Status':>10s}  {'Time':>8s}  {'Pixels':>8s}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}")
    for r in results:
        print(f"  {r['sample_idx']:8d}  {r['label']:6d}  {r['epsilon']:8.4f}  "
              f"{r['status']:>10s}  {r['elapsed']:7.1f}s  {r['n_active_pixels']:8d}")

    total_time = sum(r["elapsed"] for r in results)
    print(f"\n  Total verification time: {total_time:.1f}s")
    print(f"  Results saved to {results_path}")

    # 对比原始模型
    orig_results_path = os.path.join(os.path.dirname(__file__), "verification_results.json")
    print_comparison(results, orig_results_path)

    unsafe_results = [r for r in results if r["status"] == "unsafe"]
    if unsafe_results:
        print(f"\n  Found {len(unsafe_results)} adversarial counterexamples!")
    else:
        print(f"\n  No adversarial counterexamples found by Z3.")

if __name__ == "__main__":
    main()
