import PolyLeanVerify.Basic
import Mathlib.Tactic

/-!
# 专家混合（MoE）路由形式化

对应 `arch/moe-fns.ts`。

核心安全性质：
1. **Top-k 稀疏性**：只有 k 个专家被激活（其余权重为 0）
2. **活跃占比上界**：activeFraction ≤ topK / nExperts（MoE）或 1/nExperts（Switch）
3. **Dense 模式全激活**：所有专家都被激活
-/

namespace PolyLeanVerify

/-- MoE 变体 -/
inductive MoEVariant where
  | dense    -- 全部专家激活
  | moe      -- Top-k 路由
  | switch   -- Top-1 路由
  | deepseek -- Top-k + 共享专家
  deriving Repr

/-- 门控分数：用 token 值与专家索引算 [0,1] 分数
    对应 `gateScores(token, nExperts)` -/
def gateScores (token : Float) (nExperts : Nat) : Vec :=
  (List.range nExperts).map (fun e =>
    Float.sin (token * 0.7 + e.toFloat * 1.3) * 0.5 + 0.5)

/-- 分数最高的 k 个专家索引（降序）
    对应 `topKExperts(scores, k)` -/
def topKExperts (scores : Vec) (k : Nat) : List Nat :=
  let indexed := scores.mapIdx (fun i v => (v, i))
  let sorted := indexed.toArray.qsort (fun a b => a.1 > b.1)
  (sorted.toList.take k).map (fun (_, i) => i)

/-- 每个 token 激活的专家占比（稀疏度）
    对应 `activeFraction(variant, nExperts, topK, nShared)` -/
def activeFraction (variant : MoEVariant) (nExperts : Nat) (topK : Nat) (nShared : Nat) : Float :=
  match variant with
  | MoEVariant.dense => 1.0
  | MoEVariant.switch => 1.0 / nExperts.toFloat
  | MoEVariant.moe => topK.toFloat / nExperts.toFloat
  | MoEVariant.deepseek => (topK.toFloat + nShared.toFloat) / (nExperts.toFloat + nShared.toFloat)

/-- 被路由激活的专家索引列表
    对应 `routedExperts(variant, token, nExperts, topK)` -/
def routedExperts (variant : MoEVariant) (token : Float) (nExperts : Nat) (topK : Nat) : List Nat :=
  let scores := gateScores token nExperts
  match variant with
  | MoEVariant.dense => (List.range nExperts).toArray.toList
  | MoEVariant.switch => topKExperts scores 1
  | MoEVariant.moe => topKExperts scores topK
  | MoEVariant.deepseek => topKExperts scores topK

end PolyLeanVerify
