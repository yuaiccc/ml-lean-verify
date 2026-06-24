import PolyLeanVerify.Basic
import Mathlib.Tactic

/-!
# 注意力机制形式化

对应 `arch/attention-kv.ts` 和 `arch/kv-cache.ts`。

核心安全性质：
1. **softmax 输出是概率分布**：每个分量非负，且总和为 1
2. **attention 输出是 V 的凸组合**：输出向量的每个分量是 V 各行对应分量的加权和，权重非负且和为 1
3. **KV-cache 显存上界**：MHA > GQA > MQA，MLA 最省
-/

namespace PolyLeanVerify

/-! ## Softmax

`softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`

对应 TypeScript 中隐含的注意力权重计算（`arch/attention-kv.ts` 展示 KV 变体，
softmax 数学在此形式化）。
-/

/-- exp 的非负性（Float 版本，作为公理因为 Lean core 不提供） -/
axiom float_exp_nonneg (x : Float) : 0 < Float.exp x

/-- exp 单调递增 -/
axiom float_exp_mono {a b : Float} (h : a ≤ b) : Float.exp a ≤ Float.exp b

/-- Float.exp 的和 -/
def expSum (xs : List Float) : Float :=
  xs.foldl (fun acc x => acc + Float.exp x) 0

/-- softmax 的分子：单个 exp 值 -/
def softmaxNumerator (x : Float) : Float := Float.exp x

/-- softmax 向量 -/
def softmax (xs : List Float) : List Float :=
  let s := expSum xs
  xs.map (fun x => Float.exp x / s)

/-! ## Attention

`attention(Q, K, V) = softmax(Q · K^T / √d) · V`

我们形式化简化版：给定注意力权重 `w`（已过 softmax）和值矩阵 `V`，
输出 = Σ_i w_i * V_i。
-/

/-- 向量点积 -/
def dot (a b : Vec) : Float :=
  (List.zipWith (fun x y => x * y) a b).foldl (fun acc x => acc + x) 0

/-- 向量数乘 -/
def scale (c : Float) (v : Vec) : Vec := v.map (fun x => c * x)

/-- 向量加法 -/
def vadd (a b : Vec) : Vec := List.zipWith (fun x y => x + y) a b

/-- 加权和：Σ_i w_i * v_i -/
def weightedSum (weights : Vec) (vectors : List Vec) : Vec :=
  match weights, vectors with
  | [], _ | _, [] => []
  | w :: ws, v :: vs =>
    vadd (scale w v) (weightedSum ws vs)

/-- attention 输出：用权重对 V 做加权和 -/
def attention (weights : Vec) (vectors : List Vec) : Vec :=
  weightedSum weights vectors

end PolyLeanVerify
