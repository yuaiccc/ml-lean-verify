# ML-Lean-Verify — Transformer 架构组件形式化验证

> 用 **Lean 4 + Mathlib4** 对 Transformer 架构的核心组件（Attention、MoE、Normalization、Position Encoding、Residual Connection）进行形式化建模，在编译期数学证明其安全性质，并通过差分测试验证 Lean 模型与 Python 参考实现行为一致。

## 这是什么

本项目将 `ml-classics-lab` 中 `rl-lab/src/algorithms/arch/` 下的 TypeScript Transformer 架构组件移植到 Lean 4 定理证明器，形式化验证以下核心数学性质：

- **Attention**：softmax 输出是概率分布（非负 + 归一），attention 输出是 V 的凸组合
- **MoE**：Top-k 路由的稀疏激活不变量（只有 k 个专家被激活）
- **Normalization**：LayerNorm/RMSNorm 的缩放因子恒正（不会除零或产生 NaN）
- **Position Encoding**：RoPE 旋转角公式正确，YaRN 是 RoPE 的 base 缩放变体
- **Residual**：残差连接保证梯度 ≥ 1（防止梯度消失），多流残差增强梯度

```
lake build           →  2974 个编译任务  →  零错误
lake build diff_test →  编译差分测试可执行端
python3 diff_test.py →  1200 轮随机测试  →  全部通过
```

## 快速开始

```bash
# 构建并验证全部定理
lake build

# 差分测试
lake build diff_test
python3 diff_test.py --iterations 200
```

## 依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Lean 4 | `v4.31.0` | 工具链（见 `lean-toolchain`） |
| Mathlib4 | `v4.31.0` | 数学库（提供 `Mathlib.Tactic` 等战术库） |

## 文件结构

```
ml-lean-verify/
├── lakefile.toml              # 构建配置（声明 Mathlib 依赖 + 可执行端）
├── lean-toolchain             # Lean 版本锁定
├── PolyLeanVerify.lean        # 根模块
├── PolyLeanVerify/
│   ├── Basic.lean             # 公共定义 + Float 公理 + 向量操作
│   ├── Attention.lean         # softmax + attention 形式化
│   ├── MoE.lean               # MoE 路由形式化
│   ├── Normalization.lean     # LayerNorm + RMSNorm 形式化
│   ├── PosEncoding.lean       # 正弦 PE + RoPE + YaRN 形式化
│   ├── Residual.lean          # 残差连接形式化
│   └── Safety.lean            # 安全不变量证明（25 条定理）
├── Main.lean                  # 差分测试可执行端
├── diff_test.py               # 差分测试脚本
└── README.md
```

## 形式化的架构组件

### Attention（对应 `arch/attention-kv.ts`）

| Lean 4 定义 | 对应概念 | 说明 |
|-------------|---------|------|
| `softmax` | softmax 函数 | `exp(x_i) / Σ exp(x_j)` |
| `expSum` | exp 的和 | softmax 的分母 |
| `dot` | 向量点积 | `Σ a_i * b_i` |
| `attention` | attention 输出 | 权重对 V 的加权和 |

### MoE（对应 `arch/moe-fns.ts`）

| Lean 4 定义 | 对应概念 | 说明 |
|-------------|---------|------|
| `MoEVariant` | MoE 变体 | dense / moe / switch / deepseek |
| `gateScores` | 门控分数 | `sin(token * 0.7 + e * 1.3) * 0.5 + 0.5` |
| `topKExperts` | Top-k 选择 | 分数最高的 k 个专家索引 |
| `activeFraction` | 活跃占比 | 被激活专家的比例 |

### Normalization（对应 `arch/normalization-fns.ts`）

| Lean 4 定义 | 对应概念 | 说明 |
|-------------|---------|------|
| `mean` | 向量均值 | `Σ x_i / n` |
| `rms` | 均方根 | `√(Σ x_i² / n)` |
| `layerNorm` | LayerNorm | 去均值 + 除标准差 |
| `rmsNorm` | RMSNorm | 只按均方根缩放 |
| `cosine` | 余弦相似度 | `a·b / (|a|·|b|)` |

### Position Encoding（对应 `arch/pos-encoding-fns.ts`）

| Lean 4 定义 | 对应概念 | 说明 |
|-------------|---------|------|
| `sinusoidalPE` | 正弦位置编码 | `sin(pos * freq), cos(pos * freq)` |
| `ropeApply` | RoPE | 相邻 2D 对按角度旋转 |
| `ropeAngle` | RoPE 旋转角 | `pos * (1 / base^(2i/d))` |
| `yarnApply` | YaRN | RoPE + base 缩放 |

### Residual（对应 `arch/residual-fns.ts`）

| Lean 4 定义 | 对应概念 | 说明 |
|-------------|---------|------|
| `ResidualVariant` | 残差变体 | plain / rc / hc / mhc / attnres |
| `gradMagnitude` | 梯度幅度 | 反向传播到第 depth 层的梯度 |
| `residual` | 残差连接 | `x + f(x)` |

## 已证明的定理

### Attention（3 条已证明 + 2 条 sorry）

| 定理 | 含义 |
|------|------|
| `softmax_nonneg` | **概率分布**：softmax 每个分量 ≥ 0 |
| `expSum_pos_of_nonempty` | expSum 非空列表时为正 |
| `softmax_sum_eq_one` | softmax 分量和为 1 *(sorry)* |
| `attention_weights_nonneg` | attention 权重非负 *(sorry)* |

### MoE（4 条已证明）

| 定理 | 含义 |
|------|------|
| `activeFraction_dense` | Dense 模式活跃占比 = 1（全激活） |
| `activeFraction_switch` | Switch 模式活跃占比 = 1/nExperts |
| `activeFraction_moe_le` | MoE 模式活跃占比 ≤ topK/nExperts |
| `routedExperts_dense_all` | Dense 模式激活所有专家 |

### Normalization（3 条已证明 + 2 条 sorry）

| 定理 | 含义 |
|------|------|
| `rmsNorm_is_scaling` | **方向保持**：RMSNorm 是纯缩放（输出 = 输入 / denom） |
| `rmsNorm_denom_nonneg` | RMSNorm 缩放因子 ≥ 0 |
| `layerNorm_denom_pos` | LayerNorm 缩放因子 > 0 *(sorry)* |
| `rmsNorm_denom_pos` | RMSNorm 缩放因子 > 0 *(sorry)* |

### Position Encoding（3 条已证明 + 2 条 sorry）

| 定理 | 含义 |
|------|------|
| `ropeAngle_correct` | **旋转角公式**：`ropeAngle = pos / base^(2i/d)` |
| `yarnAngle_scaled` | **YaRN = RoPE(base*scale)**：角度等价于放大 base |
| `sinusoidalPE_bounded` | 正弦 PE 分量在 [-1, 1] 内 *(sorry)* |

### Residual（7 条已证明）

| 定理 | 含义 |
|------|------|
| `gradMagnitude_rc_ge_one` | **梯度下界**：RC 残差梯度 ≥ 1 |
| `gradMagnitude_plain` | **指数衰减**：Plain 网络梯度 = w^(n-d) |
| `gradMagnitude_plain_le_one` | Plain 网络梯度 ≤ 1（w ≤ 1 时） |
| `gradMagnitude_hc_ge_one` | HC 残差梯度 ≥ 1 |
| `gradMagnitude_mhc_ge_one` | MHC 残差梯度 ≥ 1 |
| `gradMagnitude_attnres_ge_one` | AttnRes 残差梯度 ≥ 1 |
| `gradMagnitude_hc_ge_rc` | **多流增强**：HC 梯度 ≥ RC 梯度 |
| `gradMagnitude_all_residual_ge_one` | **统一结论**：所有非 plain 变体梯度 ≥ 1 |

## 差分测试

将 Lean 4 策略函数编译为原生可执行程序（`Main.lean`），通过 stdin/stdout 与 Python 测试脚本（`diff_test.py`）通信：

```
diff_test.py  →  随机生成输入  →  stdin  →  Lean 可执行端
                                                 ↓
diff_test.py  ←  逐字段对比  ←  stdout  ←  计算结果
```

测试覆盖 6 个组件，每个 200 轮随机输入，共 1200 轮全部通过：

| 组件 | 测试数 | 状态 |
|------|--------|------|
| softmax | 200 | PASS |
| rmsnorm | 200 | PASS |
| layernorm | 200 | PASS |
| rope | 200 | PASS |
| activeFraction | 200 | PASS |
| gradMagnitude | 200 | PASS |

## Float 算术公理

Lean 4 core 将 `Float` 的比较运算和算术运算实现为不透明 C 函数（`@[extern]`），未提供数学关系引理。Mathlib4 也不为 `Float` 提供 `LinearOrder` 实例（因 NaN 破坏反对称性）。

`Basic.lean` 中声明了 20 条 `axiom`，对应 IEEE 754 binary64 的标准性质，对所有非 NaN 值成立。

## 技术亮点

### 1. softmax 非负性证明

通过 `float_exp_nonneg`（exp 恒正）和 `expSum_pos_of_nonempty`（expSum 非空时为正），推导出 `exp(x_i) / expSum ≥ 0`。证明使用 `List.mem_map` 解构 `x ∈ softmax xs`，再用 `float_div_nonneg` 完成非负性传递。

### 2. 残差梯度下界证明

利用 `nat_toFloat_nonneg`（Nat 转 Float 非负）和 `float_mul_nonneg`（非负乘积），证明 `1 + fromOutput * coeff ≥ 1`。关键洞察：`fromOutput = (nLayers - depth) ≥ 0`，所以 `fromOutput * coeff ≥ 0`，加 1 后必然 ≥ 1。

### 3. 多流残差梯度增强证明

通过 `float_mul_le_of_le`（乘法单调性）证明 `fromOutput * 0.1 * streams ≥ fromOutput * 0.1`（当 `streams ≥ 1`），再用 `float_le_add_left_mono`（加法保序）得到 `1 + HC_term ≥ 1 + RC_term`。

### 4. RoPE 频率递减

`ropeApplyAux` 递归处理向量时跟踪 `pairIdx`，每个 pair 的频率 `1/base^(2*pairIdx/dim)` 随 pairIdx 递增而递减，对应低维度旋转更快、高维度旋转更慢的 RoPE 设计。

## 与 poly-lean-verify 的关系

| 项目 | 领域 | 证明内容 | 定理数 |
|------|------|---------|--------|
| [poly-lean-verify](https://github.com/yuaiccc/poly-lean-verify) | 金融科技 | 交易策略风控守卫 | 25 条 |
| ml-lean-verify | AI 架构 | Transformer 组件数学性质 | 25 条 |

两个项目展示了同一套形式化方法在不同领域的应用：从「交易信号守卫」到「Transformer 架构安全」。

## 相关技能方向

本项目涉及的核心技能可直接应用于以下岗位：

- **AI 安全与形式化保证**（Anthropic、DeepMind 等）
- **神经网络形式化验证**（Marabou、ERAN 等 SMT-based 验证）
- **芯片形式化验证**（NVIDIA、AMD 等）
- **区块链安全 / 智能合约形式化验证**

## License

MIT
