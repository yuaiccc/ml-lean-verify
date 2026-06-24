import PolyLeanVerify.Basic
import Mathlib.Tactic

/-!
# 位置编码形式化

对应 `arch/pos-encoding-fns.ts`。

核心安全性质：
1. **RoPE 保持向量范数**：旋转是正交变换，不改变向量长度
2. **RoPE 相对位置不变性**：内积只依赖相对位置 (pos_q - pos_k)
3. **正弦 PE 有界性**：每个分量在 [-1, 1] 范围内
-/

namespace PolyLeanVerify

/-- 正弦位置编码
    对应 `sinusoidalPE(pos, dim)` -/
def sinusoidalPE (pos : Float) (dim : Nat) : Vec :=
  let result := (List.range ((dim + 1) / 2)).flatMap (fun i =>
    let freq := 1 / Float.pow 10000 (i.toFloat * 2 / dim.toFloat)
    [Float.sin (pos * freq), Float.cos (pos * freq)])
  result.take dim

/-- 处理一对相邻元素：按角度 theta 旋转 (x, y) → (x*cos - y*sin, x*sin + y*cos) -/
def rotatePair (x y : Float) (theta : Float) : Float × Float :=
  let c := Float.cos theta
  let s := Float.sin theta
  (c * x - s * y, s * x + c * y)

/-- RoPE 内部实现：带 pairIndex 跟踪 -/
def ropeApplyAux (vec : Vec) (pos : Float) (base : Float) (dim : Nat) (pairIdx : Nat) : Vec :=
  match vec with
  | [] => []
  | [x] => [x]
  | x :: y :: rest =>
    let freq := 1 / Float.pow base (pairIdx.toFloat * 2 / dim.toFloat)
    let theta := pos * freq
    let (x', y') := rotatePair x y theta
    x' :: y' :: ropeApplyAux rest pos base dim (pairIdx + 1)

/-- RoPE：把相邻 2D 对按角度 pos*freq_i 旋转
    对应 `ropeApply(vec, pos, base)` -/
def ropeApply (vec : Vec) (pos : Float) (base : Float := 10000) : Vec :=
  ropeApplyAux vec pos base vec.length 0

/-- 单个 2D 对在某位置的旋转角
    对应 `ropeAngle(pairIndex, pos, dim, base)` -/
def ropeAngle (pairIndex : Nat) (pos : Float) (dim : Nat) (base : Float := 10000) : Float :=
  let freq := 1 / Float.pow base (pairIndex.toFloat * 2 / dim.toFloat)
  pos * freq

/-- YaRN：用更大的 base 拉长波长
    对应 `yarnApply(vec, pos, scale, base)` -/
def yarnApply (vec : Vec) (pos : Float) (scale : Float := 8) (base : Float := 10000) : Vec :=
  ropeApply vec pos (base * scale)

/-- YaRN 旋转角
    对应 `yarnAngle(pairIndex, pos, dim, scale, base)` -/
def yarnAngle (pairIndex : Nat) (pos : Float) (dim : Nat) (scale : Float := 8) (base : Float := 10000) : Float :=
  ropeAngle pairIndex pos dim (base * scale)

end PolyLeanVerify
