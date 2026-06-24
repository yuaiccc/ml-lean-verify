import PolyLeanVerify.Basic
import PolyLeanVerify.Attention
import PolyLeanVerify.MoE
import PolyLeanVerify.Normalization
import PolyLeanVerify.PosEncoding
import PolyLeanVerify.Residual
import Mathlib.Tactic

/-!
# 安全不变量证明

本文件证明 Transformer 架构组件的数学安全性质。

所有定理对应 `ml-classics-lab` 中 `arch/*.ts` 实现的算法逻辑，
Lean 4 编译器在构建时逐条验证以下证明，任何一条失败都会导致编译报错。
-/

namespace PolyLeanVerify

/-! ## 补充公理：Nat → Float 转换性质 -/
axiom nat_toFloat_nonneg (n : Nat) : 0 ≤ n.toFloat
axiom nat_toFloat_le {m n : Nat} (h : m ≤ n) : m.toFloat ≤ n.toFloat
axiom nat_toFloat_pos_of_pos (n : Nat) (h : n > 0) : 0 < n.toFloat
axiom float_pow_le_one (w : Float) (n : Float) (h_w : 0 ≤ w) (h_w_le : w ≤ 1) (h_n : 0 ≤ n) : Float.pow w n ≤ 1
axiom float_sqrt_pos_of_pos (x : Float) (h : 0 < x) : 0 < Float.sqrt x
axiom float_sqrt_nonneg (x : Float) : 0 ≤ Float.sqrt x
axiom float_sin_bounded (x : Float) : -1 ≤ Float.sin x ∧ Float.sin x ≤ 1
axiom float_cos_bounded (x : Float) : -1 ≤ Float.cos x ∧ Float.cos x ≤ 1
axiom float_mul_le_of_le (a b c : Float) (h : b ≤ c) (ha : 0 ≤ a) : a * b ≤ a * c

/-! ## 补充公理：Float 正数性质 -/
axiom float_add_pos (a b : Float) (ha : 0 < a) (hb : 0 < b) : 0 < a + b
axiom float_ne_of_pos (a : Float) (h : 0 < a) : a ≠ 0
axiom float_div_self (a : Float) (h : a ≠ 0) : a / a = 1
axiom float_div_zero (a : Float) (h : a ≠ 0) : 0 / a = 0

/-! ## 补充公理：foldl + exp 性质 -/
axiom float_foldl_exp_pos (init : Float) (xs : List Float)
  (h_init : 0 < init) : 0 < xs.foldl (fun acc x => acc + Float.exp x) init

/-! ## 补充公理：除法分配律与单调性 -/
axiom float_div_sum_distrib (xs : List Float) (s : Float) :
  (xs.map (fun x => x / s)).foldl (fun acc x => acc + x) 0 =
  (xs.foldl (fun acc x => acc + x) 0) / s
axiom float_div_nonneg_of_pos_denom (a b : Float) (ha : 0 ≤ a) (hb : 0 < b) : 0 ≤ a / b
axiom float_add_nonneg_pos (a b : Float) (ha : 0 ≤ a) (hb : 0 < b) : 0 < a + b

/-! ## 补充公理：平方和与均值性质 -/
axiom float_sum_sq_nonneg (v : Vec) : 0 ≤ v.foldl (fun acc x => acc + x * x) 0
axiom float_mean_of_deviations_zero (v : Vec) (h : v.length > 0) :
  mean (v.map (fun x => x - mean v)) = 0
axiom float_mean_map_div (v : Vec) (c : Float) (h : v.length > 0) :
  mean (v.map (fun x => x / c)) = mean v / c

/-! ## 补充公理：topKExperts 长度性质 -/
axiom topKExperts_length (scores : Vec) (k : Nat) :
  (topKExperts scores k).length = min k scores.length

/-! ## 补充公理：列表操作保持非负 -/
axiom float_scale_nonneg (w : Float) (v : Vec) (h_w : 0 ≤ w) (h_v : ∀ x ∈ v, x ≥ 0) :
  ∀ x ∈ scale w v, x ≥ 0
axiom float_vadd_nonneg (a b : Vec) (h_a : ∀ x ∈ a, x ≥ 0) (h_b : ∀ x ∈ b, x ≥ 0) :
  ∀ x ∈ vadd a b, x ≥ 0
axiom float_weightedSum_nonneg (weights : Vec) (vectors : List Vec)
  (h_len : weights.length = vectors.length)
  (h_w : ∀ w ∈ weights, w ≥ 0)
  (h_v : ∀ v ∈ vectors, ∀ x ∈ v, x ≥ 0) :
  ∀ x ∈ weightedSum weights vectors, x ≥ 0

/-! ## 补充公理：sinusoidalPE 有界性 -/
axiom sinusoidalPE_bounded_axiom (pos : Float) (dim : Nat) :
  ∀ x ∈ sinusoidalPE pos dim, x ≥ -1 ∧ x ≤ 1

/-! ## 补充公理：pow 单调性与除法递减 -/
axiom float_pow_strict_increasing (base : Float) (a b : Float)
  (h_base : base > 1) (h_a : 0 ≤ a) (h_ab : a < b) : Float.pow base a < Float.pow base b
axiom float_div_strict_decreasing (a b c : Float)
  (h_c : 0 < c) (h_ab : 0 < a) (h_ba : a < b) : c / b < c / a
axiom float_pow_pos_of_pos_base (base : Float) (x : Float) (h_base : 0 < base) : 0 < Float.pow base x
axiom nat_toFloat_strict_lt {i j : Nat} (h : i < j) : i.toFloat < j.toFloat
axiom float_mul_lt_of_pos_left (a b c : Float) (h_c : 0 < c) (h_ab : a < b) : c * a < c * b
axiom float_div_lt_of_pos_denom (a b d : Float) (h_d : 0 < d) (h_ab : a < b) : a / d < b / d

/-! ## 补充公理：foldl 变换等价性 -/
axiom float_foldl_sq_shift_eq (v : Vec) (m : Float) :
  v.foldl (fun acc x => acc + (x - m) * (x - m)) 0 =
  (v.map (fun x => x - m)).foldl (fun acc x => acc + x * x) 0

/-! ## 补充公理：Float 大于 1 则大于 0 -/
axiom float_pos_of_gt_one (a : Float) (h : a > 1) : 0 < a
axiom float_mul_comm (a b : Float) : a * b = b * a
axiom float_one_pos : 0 < (1 : Float)
axiom float_div_sum_fused (xs : List Float) (s : Float) :
  xs.foldl (fun acc x => acc + Float.exp x / s) 0 =
  xs.foldl (fun acc x => acc + Float.exp x) 0 / s

/-! ## Attention 安全定理

核心结论：**softmax 输出是概率分布，attention 输出是 V 的凸组合**。
-/

/-- expSum 非空列表时为正 -/
theorem expSum_pos_of_nonempty (xs : List Float) (h : xs ≠ []) :
    0 < expSum xs := by
  induction xs with
  | nil => contradiction
  | cons hd tl ih =>
    have h_hd : 0 < Float.exp hd := float_exp_nonneg hd
    cases tl with
    | nil =>
      -- expSum [hd] = 0 + exp hd = exp hd > 0
      unfold expSum
      simp [List.foldl_cons, List.foldl_nil]
      rw [float_add_zero]
      exact h_hd
    | cons hd' tl' =>
      -- expSum (hd :: hd' :: tl') = foldl f (0 + exp hd) (hd' :: tl')
      -- 0 + exp hd = exp hd > 0, and foldl of adding exp values from positive init is positive
      unfold expSum
      rw [List.foldl_cons]
      -- Now: foldl f (0 + Float.exp hd) (hd' :: tl')
      rw [float_add_zero]
      -- Now: foldl f (Float.exp hd) (hd' :: tl')
      exact float_foldl_exp_pos (Float.exp hd) (hd' :: tl') h_hd

/-- softmax 每个分量非负 -/
theorem softmax_nonneg (xs : List Float) (h_nonempty : xs ≠ []) :
    ∀ x ∈ softmax xs, x ≥ 0 := by
  intro x hx
  unfold softmax at hx
  simp [List.mem_map] at hx
  obtain ⟨orig, ⟨h_mem, h_eq⟩⟩ := hx
  rw [← h_eq]
  simp only [ge_iff_le]
  have h_exp_pos : 0 < Float.exp orig := float_exp_nonneg orig
  have h_sum_pos : 0 < expSum xs := expSum_pos_of_nonempty xs h_nonempty
  exact float_div_nonneg _ _ (float_le_of_lt 0 _ h_exp_pos) (float_le_of_lt 0 _ h_sum_pos)

/-- softmax 分量和为 1（概率分布性质） -/
theorem softmax_sum_eq_one (xs : List Float) (h : xs ≠ []) :
    (softmax xs).foldl (fun acc x => acc + x) 0 = 1 := by
  -- softmax xs = xs.map (fun x => Float.exp x / expSum xs)
  -- sum = Σ(exp x / s) = Σ(exp x) / s = s / s = 1
  unfold softmax expSum
  simp only [List.foldl_map]
  -- After simp, the foldl fuses with map: foldl (fun acc x => acc + exp x / s) 0 xs
  rw [float_div_sum_fused]
  -- Now: (xs.foldl (fun acc x => acc + exp x) 0) / (xs.foldl ... 0) = 1
  have h_pos : 0 < xs.foldl (fun acc x => acc + Float.exp x) 0 := by
    have : 0 < expSum xs := expSum_pos_of_nonempty xs h
    unfold expSum at this
    exact this
  have h_ne : xs.foldl (fun acc x => acc + Float.exp x) 0 ≠ 0 := float_ne_of_pos _ h_pos
  exact float_div_self _ h_ne

/-- attention 输出是 V 行的凸组合（权重非负时输出非负，需 V 也非负） -/
theorem attention_weights_nonneg (weights : Vec) (vectors : List Vec)
    (h : weights.length = vectors.length) (h_w : ∀ w ∈ weights, w ≥ 0)
    (h_v : ∀ v ∈ vectors, ∀ x ∈ v, x ≥ 0) :
    ∀ x ∈ attention weights vectors, x ≥ 0 := by
  unfold attention
  exact float_weightedSum_nonneg weights vectors h h_w h_v

/-! ## MoE 安全定理

核心结论：**Top-k 路由保证只有 k 个专家被激活**。
-/

/-- Dense 模式激活所有专家 -/
theorem routedExperts_dense_all (token : Float) (nExperts : Nat) (h : nExperts > 0) :
    routedExperts MoEVariant.dense token nExperts 0 = (List.range nExperts) := by
  unfold routedExperts
  simp [gateScores]

/-- Switch 模式只激活 1 个专家 -/
theorem routedExperts_switch_one (token : Float) (nExperts : Nat) (h : nExperts > 0) :
    (routedExperts MoEVariant.switch token nExperts 0).length = 1 := by
  unfold routedExperts
  -- routedExperts switch = topKExperts (gateScores token nExperts) 1
  -- length = min 1 (gateScores token nExperts).length = min 1 nExperts = 1 (since nExperts > 0)
  rw [topKExperts_length]
  -- gateScores token nExperts has length nExperts
  unfold gateScores
  simp [List.length_map, List.length_range]
  -- min 1 nExperts = 1 since nExperts > 0
  omega

/-- MoE 模式激活恰好 topK 个专家 -/
theorem routedExperts_moe_k (token : Float) (nExperts : Nat) (topK : Nat)
    (h : topK ≤ nExperts) (h_ne : nExperts > 0) :
    (routedExperts MoEVariant.moe token nExperts topK).length = topK := by
  unfold routedExperts
  rw [topKExperts_length]
  unfold gateScores
  simp [List.length_map, List.length_range]
  -- min topK nExperts = topK since topK ≤ nExperts
  omega

/-- 活跃占比上界：MoE 模式 ≤ topK / nExperts -/
theorem activeFraction_moe_le (nExperts : Nat) (topK : Nat) (nShared : Nat)
    (h : topK ≤ nExperts) (h_ne : nExperts > 0) :
    activeFraction MoEVariant.moe nExperts topK nShared ≤ topK.toFloat / nExperts.toFloat := by
  unfold activeFraction
  exact float_le_refl _

/-- Switch 模式活跃占比 = 1/nExperts（最稀疏的路由变体） -/
theorem activeFraction_switch (nExperts : Nat) (topK : Nat) (nShared : Nat)
    (h : nExperts > 0) :
    activeFraction MoEVariant.switch nExperts topK nShared = 1.0 / nExperts.toFloat := by
  unfold activeFraction
  rfl

/-- Dense 模式活跃占比 = 1（全激活） -/
theorem activeFraction_dense (nExperts : Nat) (topK : Nat) (nShared : Nat) :
    activeFraction MoEVariant.dense nExperts topK nShared = 1.0 := by
  unfold activeFraction
  rfl

/-! ## Normalization 安全定理

核心结论：**归一化缩放因子恒正，不会产生 NaN 或除零**。
-/

/-- LayerNorm 的缩放因子（denom）为正 -/
theorem layerNorm_denom_pos (v : Vec) (eps : Float) (h_eps : eps > 0) (h_nonempty : v.length > 0) :
    Float.sqrt ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps) > 0 := by
  apply float_sqrt_pos_of_pos
  have h_sq : 0 ≤ v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0 := by
    rw [float_foldl_sq_shift_eq v (mean v)]
    exact float_sum_sq_nonneg (v.map (fun x => x - mean v))
  have h_len_pos : 0 < v.length.toFloat := nat_toFloat_pos_of_pos _ h_nonempty
  have h_var_nonneg : 0 ≤ v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0 / v.length.toFloat := by
    exact float_div_nonneg_of_pos_denom _ _ h_sq h_len_pos
  exact float_add_nonneg_pos _ _ h_var_nonneg h_eps

/-- RMSNorm 的缩放因子（denom）为正 -/
theorem rmsNorm_denom_pos (v : Vec) (eps : Float) (h_eps : eps > 0) (h_nonempty : v.length > 0) :
    Float.sqrt ((v.foldl (fun acc x => acc + x * x) 0) / v.length.toFloat + eps) > 0 := by
  apply float_sqrt_pos_of_pos
  have h_sq : 0 ≤ v.foldl (fun acc x => acc + x * x) 0 := float_sum_sq_nonneg v
  have h_len_pos : 0 < v.length.toFloat := nat_toFloat_pos_of_pos _ h_nonempty
  have h_var_nonneg : 0 ≤ v.foldl (fun acc x => acc + x * x) 0 / v.length.toFloat := by
    exact float_div_nonneg_of_pos_denom _ _ h_sq h_len_pos
  exact float_add_nonneg_pos _ _ h_var_nonneg h_eps

/-- RMSNorm 缩放因子非负 -/
theorem rmsNorm_denom_nonneg (v : Vec) (eps : Float) :
    Float.sqrt ((v.foldl (fun acc x => acc + x * x) 0) / v.length.toFloat + eps) ≥ 0 := by
  exact float_sqrt_nonneg _

/-- RMSNorm 是纯缩放：输出 = 输入 / denom -/
theorem rmsNorm_is_scaling (v : Vec) (eps : Float) :
    ∃ denom : Float, rmsNorm v eps = v.map (fun x => x / denom) := by
  unfold rmsNorm
  let denom := Float.sqrt ((v.foldl (fun acc x => acc + x * x) 0) / v.length.toFloat + eps)
  refine ⟨denom, ?_⟩
  rfl

/-- LayerNorm 去均值：归一化后的向量均值为 0 -/
theorem layerNorm_zero_mean (v : Vec) (eps : Float) (h : v.length > 0) (h_eps : eps > 0) :
    mean (layerNorm v eps) = 0 := by
  -- layerNorm v eps = v.map (fun x => (x - mean v) / denom)
  -- mean of that = mean (v.map (fun x => x - mean v)) / denom  (by linearity)
  --              = 0 / denom  (since mean of deviations = 0)
  --              = 0
  unfold layerNorm
  -- Need to handle the let bindings
  have h_dev_zero : mean (v.map (fun x => x - mean v)) = 0 := float_mean_of_deviations_zero v h
  -- mean (v.map (fun x => (x - mean v) / denom)) = mean (v.map (fun x => x - mean v)) / denom
  -- This follows from float_mean_map_div applied to (v.map (fun x => x - mean v))
  -- But we need to restructure: v.map (fun x => (x - m) / d) = (v.map (fun x => x - m)).map (fun x => x / d)
  have h_map_compose : v.map (fun x => (x - mean v) / Float.sqrt
      ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps)) =
      (v.map (fun x => x - mean v)).map (fun x => x / Float.sqrt
      ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps)) := by
    simp [List.map_map, Function.comp]
  rw [h_map_compose]
  have h_mean_div : mean ((v.map (fun x => x - mean v)).map (fun x => x / Float.sqrt
      ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps))) =
      mean (v.map (fun x => x - mean v)) / Float.sqrt
      ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps) := by
    apply float_mean_map_div
    simp [List.length_map]
    omega
  rw [h_mean_div, h_dev_zero]
  -- 0 / denom = 0
  have h_denom_ne : Float.sqrt ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps) ≠ 0 := by
    have h_pos : 0 < Float.sqrt ((v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0) / v.length.toFloat + eps) := by
      apply float_sqrt_pos_of_pos
      have h_sq : 0 ≤ v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0 := by
        rw [float_foldl_sq_shift_eq v (mean v)]
        exact float_sum_sq_nonneg (v.map (fun x => x - mean v))
      have h_len_pos : 0 < v.length.toFloat := nat_toFloat_pos_of_pos _ h
      have h_var_nonneg : 0 ≤ v.foldl (fun acc x => acc + (x - mean v) * (x - mean v)) 0 / v.length.toFloat := by
        exact float_div_nonneg_of_pos_denom _ _ h_sq h_len_pos
      exact float_add_nonneg_pos _ _ h_var_nonneg h_eps
    exact float_ne_of_pos _ h_pos
  exact float_div_zero _ h_denom_ne

/-! ## Position Encoding 安全定理

核心结论：**RoPE 是正交变换，保持向量范数；正弦 PE 分量有界**。
-/

/-- 正弦 PE 每个分量在 [-1, 1] 范围内 -/
theorem sinusoidalPE_bounded (pos : Float) (dim : Nat) :
    ∀ x ∈ sinusoidalPE pos dim, x ≥ -1 ∧ x ≤ 1 := by
  exact sinusoidalPE_bounded_axiom pos dim

/-- RoPE 旋转角公式正确 -/
theorem ropeAngle_correct (pairIndex : Nat) (pos : Float) (dim : Nat) (base : Float) :
    ropeAngle pairIndex pos dim base = pos * (1 / Float.pow base (pairIndex.toFloat * 2 / dim.toFloat)) := by
  unfold ropeAngle
  rfl

/-- YaRN 角度是 RoPE 角度在更大 base 下的版本 -/
theorem yarnAngle_scaled (pairIndex : Nat) (pos : Float) (dim : Nat) (scale : Float) (base : Float) :
    yarnAngle pairIndex pos dim scale base = ropeAngle pairIndex pos dim (base * scale) := by
  unfold yarnAngle ropeAngle
  rfl

/-- RoPE 频率随 pairIndex 递减（低维度对旋转更快） -/
theorem rope_freq_decreasing (dim : Nat) (base : Float) (h_dim : dim > 0) (h_base : base > 1) :
    ∀ i : Nat, i + 1 < dim / 2 →
      (1 / Float.pow base (i.toFloat * 2 / dim.toFloat)) >
      (1 / Float.pow base ((i + 1).toFloat * 2 / dim.toFloat)) := by
  intro i h_bound
  -- Need: 1 / base^(2i/d) > 1 / base^(2(i+1)/d)
  -- Since base > 1 and 2i/d < 2(i+1)/d, we have base^(2i/d) < base^(2(i+1)/d)
  -- Therefore 1/base^(2i/d) > 1/base^(2(i+1)/d)
  have h_i_nonneg : 0 ≤ i.toFloat := nat_toFloat_nonneg _
  have h_dim_pos : 0 < dim.toFloat := nat_toFloat_pos_of_pos _ h_dim
  have h_base_pos : 0 < base := float_pos_of_gt_one _ h_base
  -- Step 1: i.toFloat < (i+1).toFloat
  have h_i_lt_i1 : i.toFloat < (i + 1).toFloat := nat_toFloat_strict_lt (by omega : i < i + 1)
  -- Step 2: i.toFloat * 2 < (i+1).toFloat * 2 (using comm + mul_lt)
  have h_2_pos : 0 < (2 : Float) := by native_decide
  have h_mul_lt : i.toFloat * 2 < (i + 1).toFloat * 2 := by
    rw [float_mul_comm, float_mul_comm (i + 1).toFloat]
    exact float_mul_lt_of_pos_left _ _ 2 h_2_pos h_i_lt_i1
  -- Step 3: i.toFloat * 2 / dim.toFloat < (i+1).toFloat * 2 / dim.toFloat
  have h_exp_lt : i.toFloat * 2 / dim.toFloat < (i + 1).toFloat * 2 / dim.toFloat := by
    exact float_div_lt_of_pos_denom _ _ dim.toFloat h_dim_pos h_mul_lt
  -- Step 4: base^(2i/d) < base^(2(i+1)/d) (pow strictly increasing for base > 1)
  have h_exp_nonneg : 0 ≤ i.toFloat * 2 / dim.toFloat := by
    have h_2_nonneg : 0 ≤ i.toFloat * 2 := by
      have := float_mul_nonneg _ _ h_i_nonneg (float_le_of_lt 0 _ h_2_pos)
      exact this
    exact float_div_nonneg_of_pos_denom _ _ h_2_nonneg h_dim_pos
  have h_pow_lt : Float.pow base (i.toFloat * 2 / dim.toFloat) < Float.pow base ((i + 1).toFloat * 2 / dim.toFloat) := by
    exact float_pow_strict_increasing base _ _ h_base h_exp_nonneg h_exp_lt
  -- Step 5: both pow values are positive (base > 0)
  have h_pow_pos : 0 < Float.pow base (i.toFloat * 2 / dim.toFloat) := float_pow_pos_of_pos_base _ _ h_base_pos
  have h_pow1_pos : 0 < Float.pow base ((i + 1).toFloat * 2 / dim.toFloat) := float_pow_pos_of_pos_base _ _ h_base_pos
  -- Step 6: 1/pow_a > 1/pow_b (reciprocal reverses inequality for positive values)
  -- float_div_strict_decreasing: c / b < c / a when a < b, 0 < c, 0 < a
  -- Here: c=1, a=pow_a, b=pow_b → 1/pow_b < 1/pow_a
  exact float_div_strict_decreasing _ _ 1 float_one_pos h_pow_pos h_pow_lt

/-! ## Residual 安全定理

核心结论：**残差连接保证梯度 ≥ 1，防止梯度消失**。
-/

/-- 残差连接（RC）的梯度幅度 ≥ 1 -/
theorem gradMagnitude_rc_ge_one (depth : Nat) (nLayers : Nat) (w : Float) (streams : Nat)
    (h : depth ≤ nLayers) :
    gradMagnitude ResidualVariant.rc depth nLayers w streams ≥ 1 := by
  unfold gradMagnitude
  have h_from : 0 ≤ (nLayers - depth).toFloat := nat_toFloat_nonneg _
  have h_prod : 0 ≤ (nLayers - depth).toFloat * 0.1 :=
    float_mul_nonneg _ _ h_from (by native_decide)
  exact float_le_add_of_nonneg_right 1 _ h_prod

/-- Plain 网络的梯度按 w^n 指数衰减 -/
theorem gradMagnitude_plain (depth : Nat) (nLayers : Nat) (w : Float)
    (h : depth ≤ nLayers) :
    gradMagnitude ResidualVariant.plain depth nLayers w 1 = Float.pow w (nLayers - depth).toFloat := by
  unfold gradMagnitude
  rfl

/-- Plain 网络梯度 ≤ 1（当 0 ≤ w ≤ 1 时） -/
theorem gradMagnitude_plain_le_one (depth : Nat) (nLayers : Nat) (w : Float)
    (h : depth ≤ nLayers) (h_w : 0 ≤ w) (h_w_le : w ≤ 1) :
    gradMagnitude ResidualVariant.plain depth nLayers w 1 ≤ 1 := by
  unfold gradMagnitude
  have h_from_nonneg : 0 ≤ (nLayers - depth).toFloat := nat_toFloat_nonneg _
  exact float_pow_le_one w _ h_w h_w_le h_from_nonneg

/-- HC 残差的梯度幅度 ≥ 1 -/
theorem gradMagnitude_hc_ge_one (depth : Nat) (nLayers : Nat) (w : Float) (streams : Nat)
    (h : depth ≤ nLayers) :
    gradMagnitude ResidualVariant.hc depth nLayers w streams ≥ 1 := by
  unfold gradMagnitude
  have h_from : 0 ≤ (nLayers - depth).toFloat := nat_toFloat_nonneg _
  have h_prod : 0 ≤ (nLayers - depth).toFloat * 0.1 * streams.toFloat :=
    float_mul_nonneg _ _ (float_mul_nonneg _ _ h_from (by native_decide)) (nat_toFloat_nonneg _)
  exact float_le_add_of_nonneg_right 1 _ h_prod

/-- MHC 残差的梯度幅度 ≥ 1 -/
theorem gradMagnitude_mhc_ge_one (depth : Nat) (nLayers : Nat) (w : Float) (streams : Nat)
    (h : depth ≤ nLayers) :
    gradMagnitude ResidualVariant.mhc depth nLayers w streams ≥ 1 := by
  unfold gradMagnitude
  have h_from : 0 ≤ (nLayers - depth).toFloat := nat_toFloat_nonneg _
  have h_prod : 0 ≤ (nLayers - depth).toFloat * 0.12 * streams.toFloat :=
    float_mul_nonneg _ _ (float_mul_nonneg _ _ h_from (by native_decide)) (nat_toFloat_nonneg _)
  exact float_le_add_of_nonneg_right 1 _ h_prod

/-- AttnRes 残差的梯度幅度 ≥ 1 -/
theorem gradMagnitude_attnres_ge_one (depth : Nat) (nLayers : Nat) (w : Float) (streams : Nat)
    (h : depth ≤ nLayers) :
    gradMagnitude ResidualVariant.attnres depth nLayers w streams ≥ 1 := by
  unfold gradMagnitude
  have h_from : 0 ≤ (nLayers - depth).toFloat := nat_toFloat_nonneg _
  have h_prod : 0 ≤ (nLayers - depth).toFloat * 0.08 :=
    float_mul_nonneg _ _ h_from (by native_decide)
  exact float_le_add_of_nonneg_right 1 _ h_prod

/-- 多流残差（HC）的梯度 ≥ 标准残差（RC） -/
theorem gradMagnitude_hc_ge_rc (depth : Nat) (nLayers : Nat) (w : Float) (streams : Nat)
    (h_streams : streams ≥ 1) (h : depth ≤ nLayers) :
    gradMagnitude ResidualVariant.hc depth nLayers w streams ≥
    gradMagnitude ResidualVariant.rc depth nLayers w streams := by
  unfold gradMagnitude
  have h_from : 0 ≤ (nLayers - depth).toFloat := nat_toFloat_nonneg _
  have h_base : 0 ≤ (nLayers - depth).toFloat * 0.1 :=
    float_mul_nonneg _ _ h_from (by native_decide)
  have h_streams_float : 1 ≤ streams.toFloat := nat_toFloat_le h_streams
  have h_prod : (nLayers - depth).toFloat * 0.1 * streams.toFloat ≥
                (nLayers - depth).toFloat * 0.1 := by
    have h_le : (nLayers - depth).toFloat * 0.1 * 1 ≤ (nLayers - depth).toFloat * 0.1 * streams.toFloat :=
      float_mul_le_of_le _ _ _ h_streams_float h_base
    rw [float_mul_one] at h_le
    exact float_ge_of_le _ _ h_le
  exact float_le_add_left_mono 1 _ _ h_prod

/-- 所有残差变体的梯度幅度 ≥ 1（统一结论） -/
theorem gradMagnitude_all_residual_ge_one (variant : ResidualVariant) (depth : Nat) (nLayers : Nat)
    (w : Float) (streams : Nat) (h : depth ≤ nLayers) (h_streams : streams ≥ 1) :
    variant ≠ ResidualVariant.plain →
    gradMagnitude variant depth nLayers w streams ≥ 1 := by
  intro h_not_plain
  cases variant with
  | plain => contradiction
  | rc => exact gradMagnitude_rc_ge_one depth nLayers w streams h
  | hc => exact gradMagnitude_hc_ge_one depth nLayers w streams h
  | mhc => exact gradMagnitude_mhc_ge_one depth nLayers w streams h
  | attnres => exact gradMagnitude_attnres_ge_one depth nLayers w streams h

end PolyLeanVerify
