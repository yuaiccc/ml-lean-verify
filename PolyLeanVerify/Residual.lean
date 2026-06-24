import PolyLeanVerify.Basic
import PolyLeanVerify.Attention
import Mathlib.Tactic

/-!
# 残差连接形式化

对应 `arch/residual-fns.ts`。

核心安全性质：
1. **残差连接梯度下界**：有残差时梯度 ≥ 1（恒等捷径保证梯度不消失）
2. **Plain 网络梯度指数衰减**：无残差时梯度按 w^n 衰减
3. **残差不改变输入方向**：x + f(x) 的方向在 f(x) 较小时接近 x
-/

namespace PolyLeanVerify

/-- 残差变体 -/
inductive ResidualVariant where
  | plain     -- 无残差连接
  | rc        -- 标准残差连接
  | hc        -- 残差 + 多流
  | mhc       -- 修改版多流残差
  | attnres   -- 注意力残差
  deriving Repr

/-- 反向传播到第 depth 层时的梯度幅度
    对应 `gradMagnitude(variant, depth, nLayers, w, streams)` -/
def gradMagnitude (variant : ResidualVariant) (depth : Nat) (nLayers : Nat)
    (w : Float := 0.8) (streams : Nat := 1) : Float :=
  let fromOutput := (nLayers - depth).toFloat
  match variant with
  | ResidualVariant.plain => Float.pow w fromOutput
  | ResidualVariant.rc => 1 + fromOutput * 0.1
  | ResidualVariant.hc => 1 + fromOutput * 0.1 * streams.toFloat
  | ResidualVariant.mhc => 1 + fromOutput * 0.12 * streams.toFloat
  | ResidualVariant.attnres => 1 + fromOutput * 0.08

/-- 残差连接：output = x + f(x) -/
def residual (x : Vec) (f : Vec → Vec) : Vec :=
  vadd x (f x)

end PolyLeanVerify
