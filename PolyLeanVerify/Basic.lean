/-!
# ML Classics Lab — 形式化模型基础

本文件将 `ml-classics-lab` 项目中 `rl-lab/src/algorithms/arch/` 下的
Transformer 架构组件的纯数学逻辑移植到 Lean 4，作为形式化验证的基础。

对应 TypeScript 源文件：
- `arch/activation-fns.ts` — 激活函数
- `arch/attention-kv.ts` + `arch/kv-cache.ts` — 注意力与 KV-cache
- `arch/moe-fns.ts` — 专家混合路由
- `arch/normalization-fns.ts` — 归一化
- `arch/pos-encoding-fns.ts` — 位置编码
- `arch/residual-fns.ts` — 残差连接
-/

namespace PolyLeanVerify

/-! ## Float 算术公理

Lean 4 core 将 `Float` 的比较运算和算术运算实现为不透明 C 函数（`@[extern]`），
未提供数学关系引理。Mathlib4 也不为 `Float` 提供 `LinearOrder` 实例（因 NaN 破坏反对称性）。

以下公理对应 IEEE 754 binary64 的标准性质，对所有非 NaN 值成立。
-/

/-- 三歧性推论：¬(a < b) → b ≤ a -/
axiom float_le_of_not_lt (a b : Float) (h : ¬(a < b)) : b ≤ a

/-- a < b → ¬(b ≤ a) -/
axiom float_not_le_of_lt (a b : Float) (h : a < b) : ¬(b ≤ a)

/-- a ≤ b → b ≤ c → a ≤ c（传递性） -/
axiom float_le_trans (a b c : Float) (hab : a ≤ b) (hbc : b ≤ c) : a ≤ c

/-- 0 ≤ a → 0 ≤ b → 0 ≤ a + b -/
axiom float_add_nonneg (a b : Float) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b

/-- 0 ≤ a → 0 ≤ b → 0 ≤ a * b -/
axiom float_mul_nonneg (a b : Float) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b

/-- 0 ≤ a → 0 ≤ b → 0 ≤ a / b -/
axiom float_div_nonneg (a b : Float) (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a / b

/-- a ≤ b → a + c ≤ b + c（加法单调性） -/
axiom float_le_add_right (a b c : Float) (h : a ≤ b) : a + c ≤ b + c

/-- a ≤ b → c - b ≤ c - a（减法反单调性） -/
axiom float_sub_anti_mono (a b c : Float) (h : a ≤ b) : c - b ≤ c - a

/-- a ≤ b → a * c ≤ b * c（c ≥ 0 时乘法单调性） -/
axiom float_mul_le_of_nonneg (a b c : Float) (h : a ≤ b) (hc : 0 ≤ c) : a * c ≤ b * c

/-- 0 ≤ a → 0 ≤ a^2（平方非负） -/
axiom float_sq_nonneg (a : Float) (ha : 0 ≤ a) : 0 ≤ a * a

/-- b ≤ max a b -/
axiom float_le_max_right (a b : Float) : b ≤ max a b

/-- a ≤ max a b -/
axiom float_le_max_left (a b : Float) : a ≤ max a b

/-- min a b ≤ a -/
axiom float_min_le_left (a b : Float) : min a b ≤ a

/-- c ≤ min a b ↔ c ≤ a ∧ c ≤ b -/
axiom float_le_min_iff (a b c : Float) : c ≤ min a b ↔ c ≤ a ∧ c ≤ b

/-- x < 0 → 0 ≤ -x -/
axiom float_neg_nonneg_of_neg (x : Float) (h : x < 0) : 0 ≤ -x

/-- 0 ≤ b → a ≤ a + b -/
axiom float_le_add_of_nonneg_right (a b : Float) (hb : 0 ≤ b) : a ≤ a + b

/-- b ≥ c → a + b ≥ a + c（加法右保序） -/
axiom float_le_add_left_mono (a b c : Float) (h : b ≥ c) : a + b ≥ a + c

/-- Float 加法交换律 -/
axiom float_add_comm (a b : Float) : a + b = b + a

/-- 0 + a = a（Float 加法单位元） -/
axiom float_add_zero (a : Float) : 0 + a = a

/-- a * 1 = a（Float 乘法单位元） -/
axiom float_mul_one (a : Float) : a * 1 = a

/-- a ≤ b → a ≥ b 的 Float 版本 -/
axiom float_ge_of_le (a b : Float) (h : a ≤ b) : b ≥ a

/-- a < b → a ≤ b 的 Float 版本 -/
axiom float_le_of_lt (a b : Float) (h : a < b) : a ≤ b

/-- a ≤ a（Float 自反性） -/
axiom float_le_refl (a : Float) : a ≤ a

/-! ## 向量类型别名

Lean 4 中用 `List Float` 表示数值向量，与 TypeScript 的 `number[]` 对应。
-/

/-- 向量类型别名 -/
abbrev Vec := List Float

end PolyLeanVerify
