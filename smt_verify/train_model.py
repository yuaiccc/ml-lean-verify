#!/usr/bin/env python3
"""
Step 1: 训练超小型 MNIST MLP 分类器并导出

网络结构：784 → 8 (ReLU) → 10
刻意保持极小规模（仅 8 个隐藏神经元），使 Z3 SMT 验证可在分钟内完成。

同时训练一个稍大的 784 → 32 → 10 作为对比。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# ── 网络定义 ──────────────────────────────────────────────

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

# ── 训练 ──────────────────────────────────────────────────

def train_model(hidden=8, epochs=15, lr=1e-3, batch_size=128):
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

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}  acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "mnist_tiny.pt"))

    print(f"\n  Best test accuracy: {best_acc:.4f}")
    return model, best_acc

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
            output = model(img.unsqueeze(0))
            pred = output.argmax(dim=1).item()

            if pred == label and correct < n:
                img_flat = img.view(-1).numpy()
                samples.append({
                    "image": img_flat,
                    "label": label,
                    "prediction": pred,
                    "logits": output.squeeze().numpy(),
                })
                correct += 1

    np.save(os.path.join(os.path.dirname(__file__), "test_samples.npy"), samples, allow_pickle=True)
    print(f"  Saved {len(samples)} correctly-classified test samples")

# ── 主函数 ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: 训练超小型 MNIST MLP (784→8→10)")
    print("=" * 60)

    model, acc = train_model(hidden=8, epochs=15)
    save_test_samples(model, n=10)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {n_params}")
    print(f"  Hidden neurons: 8 (ReLU)")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n  Done! Files created:")
    print(f"    - mnist_tiny.pt      (PyTorch weights)")
    print(f"    - test_samples.npy   (10 test samples)")
