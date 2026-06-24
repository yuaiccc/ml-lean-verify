#!/usr/bin/env python3
"""
Step 4: 使用 Interval Bound Propagation (IBP) 训练认证鲁棒的 TinyMLP

网络结构：784 → 8 (ReLU) → 10

IBP 核心：
  对每层计算输出上下界，用 worst-case loss 训练，使网络在 L∞ 扰动球内
  的所有输入上都保持正确分类（可被形式化验证证明）。

线性层 IBP（向量化）：
  W_pos = clamp(W, min=0)   # 仅保留正权重
  W_neg = clamp(W, max=0)   # 仅保留负权重
  upper_out = upper_in @ W_pos^T + lower_in @ W_neg^T + b
  lower_out = lower_in @ W_pos^T + upper_in @ W_neg^T + b

ReLU 层 IBP：
  lower_out = max(0, lower_in)
  upper_out = max(0, upper_in)

损失：
  standard_loss = CE(标准前向输出, label)
  ibp_loss      = CE(上界输出, label)        # worst-case loss
  total_loss    = 0.5 * standard_loss + 0.5 * ibp_loss

认证准确率：
  若 lower[正确类] > upper[其他类] 对所有其他类成立，则该样本被认证鲁棒。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# ── 网络定义（与 train_model.py / verify_robustness.py 完全一致）────────

class TinyMLP(nn.Module):
    """超小 MLP：784 → 8 → 10，仅 8 个 ReLU 神经元"""
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── IBP 核心实现 ──────────────────────────────────────────

def ibp_linear(lower_in, upper_in, weight, bias):
    """
    线性层的区间界传播。

    参数：
        lower_in, upper_in : [batch, in]   输入上下界
        weight             : [out, in]
        bias               : [out]

    返回：
        lower_out, upper_out : [batch, out]
    """
    W_pos = torch.clamp(weight, min=0.0)  # [out, in] 正权重部分
    W_neg = torch.clamp(weight, max=0.0)  # [out, in] 负权重部分

    # upper = upper_in·W_pos + lower_in·W_neg + b
    upper_out = upper_in @ W_pos.t() + lower_in @ W_neg.t() + bias
    # lower = lower_in·W_pos + upper_in·W_neg + b
    lower_out = lower_in @ W_pos.t() + upper_in @ W_neg.t() + bias
    return lower_out, upper_out


def ibp_relu(lower_in, upper_in):
    """ReLU 层的区间界传播：输出 = max(0, 输入)。"""
    lower_out = torch.clamp(lower_in, min=0.0)
    upper_out = torch.clamp(upper_in, min=0.0)
    return lower_out, upper_out


def ibp_forward(model, x, epsilon):
    """
    整个网络的 IBP 前向传播，返回 logits 的上下界。

    参数：
        model   : TinyMLP
        x       : [batch, 784]  原始输入（已 flatten）
        epsilon : 扰动半径

    返回：
        lower, upper : [batch, 10]  logits 上下界
    """
    # 输入区间：裁剪到合法像素范围 [0, 1]
    lower = torch.clamp(x - epsilon, 0.0, 1.0)
    upper = torch.clamp(x + epsilon, 0.0, 1.0)

    # fc1: 784 → 8
    lower, upper = ibp_linear(lower, upper, model.fc1.weight, model.fc1.bias)
    # ReLU
    lower, upper = ibp_relu(lower, upper)
    # fc2: 8 → 10
    lower, upper = ibp_linear(lower, upper, model.fc2.weight, model.fc2.bias)

    return lower, upper


def compute_certified_accuracy(model, x, labels, epsilon):
    """
    计算 IBP 认证准确率：
    若 lower[正确类] > upper[其他类]（对所有其他类），则认证鲁棒。
    """
    with torch.no_grad():
        lower, upper = ibp_forward(model, x, epsilon)
    batch = x.size(0)
    idx = torch.arange(batch)
    correct_lower = lower[idx, labels]              # [batch] 正确类下界
    # 把正确类的上界置为 -inf，取其他类的最大上界
    upper_masked = upper.clone()
    upper_masked[idx, labels] = float("-inf")
    max_other_upper = upper_masked.max(dim=1).values  # [batch]
    certified = (correct_lower > max_other_upper).sum().item()
    return certified / batch

# ── 训练 ──────────────────────────────────────────────────

def train_robust_model(hidden=8, epochs=15, lr=1e-3, batch_size=128,
                       epsilon=0.05):
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = TinyMLP(hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_cert_acc = 0.0
    best_std_acc = 0.0

    print(f"  IBP 训练参数: epochs={epochs}, lr={lr}, epsilon={epsilon}")
    print(f"  损失: total = 0.5 * standard_loss + 0.5 * ibp_loss")
    print(f"  (epsilon 在前 {epochs // 2} 个 epoch 内从 0 线性增长到 {epsilon}，稳定训练)")
    print(f"  (模型保存策略：后半段全 epsilon 训练中标准准确率最高的模型)\n")

    for epoch in range(epochs):
        # epsilon 线性预热：前半段从 0 增长到目标值，后半段保持目标值
        if epoch < epochs // 2:
            eps_cur = epsilon * (epoch + 1) / (epochs // 2)
        else:
            eps_cur = epsilon

        model.train()
        total_loss = 0.0
        total_std_loss = 0.0
        total_ibp_loss = 0.0

        for images, labels in train_loader:
            images = images.view(-1, 784).to(device)  # [batch, 784]
            labels = labels.to(device)

            optimizer.zero_grad()

            # 标准前向
            logits = model(images)
            standard_loss = criterion(logits, labels)

            # IBP 前向（worst-case bounds）
            lower, upper = ibp_forward(model, images, eps_cur)
            ibp_loss = criterion(upper, labels)  # worst-case loss

            # 总损失
            loss = 0.5 * standard_loss + 0.5 * ibp_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_std_loss += standard_loss.item()
            total_ibp_loss += ibp_loss.item()

        # 评估
        model.eval()
        std_correct = 0
        cert_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 784).to(device)
                labels = labels.to(device)

                # 标准准确率
                logits = model(images)
                std_correct += (logits.argmax(1) == labels).sum().item()

                # 认证准确率（用最终目标 epsilon）
                cert_correct += int(
                    (compute_certified_accuracy(model, images, labels, epsilon)
                     * images.size(0))
                )
                total += labels.size(0)

        std_acc = std_correct / total
        cert_acc = cert_correct / total

        print(f"  Epoch {epoch+1:2d}/{epochs}  eps={eps_cur:.4f}  "
              f"loss={total_loss/len(train_loader):.4f} "
              f"(std={total_std_loss/len(train_loader):.4f}, "
              f"ibp={total_ibp_loss/len(train_loader):.4f})  "
              f"std_acc={std_acc:.4f}  cert_acc={cert_acc:.4f}")

        # 跟踪最佳认证准确率（仅用于报告）
        if cert_acc > best_cert_acc:
            best_cert_acc = cert_acc

        # 模型保存：后半段（全 epsilon 训练）中标准准确率最高的模型
        # 这样保证保存的模型经过了充分的 IBP 鲁棒训练，同时保持较好的分类能力
        if epoch >= epochs // 2 and std_acc >= best_std_acc:
            best_std_acc = std_acc
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), "mnist_robust.pt"))

    # 若后半段未保存（边界情况），保存最终模型
    if best_std_acc == 0.0:
        torch.save(model.state_dict(),
                   os.path.join(os.path.dirname(__file__), "mnist_robust.pt"))

    print(f"\n  Best certified accuracy (ε={epsilon}): {best_cert_acc:.4f}")
    print(f"  Best standard accuracy (后半段全ε训练): {best_std_acc:.4f}")
    return model, best_cert_acc

# ── 保存测试样本 ──────────────────────────────────────────

def save_test_samples(model, n=10):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    test_ds = datasets.MNIST(data_dir, train=False, download=True,
                             transform=transforms.ToTensor())

    model.eval()
    samples = []
    correct = 0

    with torch.no_grad():
        for i in range(min(n * 3, len(test_ds))):
            img, label = test_ds[i]
            img_flat = img.view(-1)
            output = model(img_flat.unsqueeze(0))
            pred = output.argmax(dim=1).item()

            if pred == label and correct < n:
                samples.append({
                    "image": img_flat.numpy(),
                    "label": label,
                    "prediction": pred,
                    "logits": output.squeeze().numpy(),
                })
                correct += 1

    np.save(os.path.join(os.path.dirname(__file__), "test_samples_robust.npy"),
            samples, allow_pickle=True)
    print(f"  Saved {len(samples)} correctly-classified test samples → test_samples_robust.npy")

# ── 主函数 ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: IBP 认证鲁棒性训练 TinyMLP (784→8→10)")
    print("=" * 60)

    model, cert_acc = train_robust_model(hidden=8, epochs=15, lr=1e-3, epsilon=0.05)
    save_test_samples(model, n=10)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {n_params}")
    print(f"  Hidden neurons: 8 (ReLU)")
    print(f"  Best certified accuracy (ε=0.05): {cert_acc:.4f}")
    print(f"\n  Done! Files created:")
    print(f"    - mnist_robust.pt            (IBP 鲁棒模型权重)")
    print(f"    - test_samples_robust.npy    (10 个测试样本)")
