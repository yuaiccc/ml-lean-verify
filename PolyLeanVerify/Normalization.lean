import PolyLeanVerify.Basic
import PolyLeanVerify.Attention
import Mathlib.Tactic

/-!
# 归一化形式化

对应 `arch/normalization-fns.ts`。

核心安全性质：
1. **LayerNorm 零均值**：归一化后向量均值为 0（非退化输入下）
2. **RMSNorm 方向保持**：只缩放不平移，输出方向与输入一致
3. **RMSNorm 非负缩放**：缩放因子恒正（denom > 0）
-/

namespace PolyLeanVerify

/-- 向量均值
    对应 `mean(v)` -/
def mean (v : Vec) : Float :=
  (v.foldl (fun acc x => acc + x) 0) / v.length.toFloat

/-- 向量均方根
    对应 `rms(v)` -/
def rms (v : Vec) : Float :=
  Float.sqrt ((v.foldl (fun acc x => acc + x * x) 0) / v.length.toFloat)

/-- LayerNorm：去均值再除以标准差
    对应 `layerNorm(v, eps)` -/
def layerNorm (v : Vec) (eps : Float := 1e-5) : Vec :=
  let m := mean v
  let variance := (v.foldl (fun acc x => acc + (x - m) * (x - m)) 0) / v.length.toFloat
  let denom := Float.sqrt (variance + eps)
  v.map (fun x => (x - m) / denom)

/-- RMSNorm：只按均方根缩放，不去均值
    对应 `rmsNorm(v, eps)` -/
def rmsNorm (v : Vec) (eps : Float := 1e-5) : Vec :=
  let ms := (v.foldl (fun acc x => acc + x * x) 0) / v.length.toFloat
  let denom := Float.sqrt (ms + eps)
  v.map (fun x => x / denom)

/-- 余弦相似度
    对应 `cosine(a, b)` -/
def cosine (a b : Vec) : Float :=
  let d := dot a b
  let na := Float.sqrt (a.foldl (fun acc x => acc + x * x) 0)
  let nb := Float.sqrt (b.foldl (fun acc x => acc + x * x) 0)
  d / (na * nb + 1e-12)

end PolyLeanVerify
