#!/usr/bin/env python3
"""
差分测试：Lean 4 形式化模型 vs Python 参考实现

测试组件：softmax, rmsnorm, layernorm, rope, activeFraction, gradMagnitude
"""

import subprocess
import random
import math
import argparse
import sys

LEAN_EXE = ".lake/build/bin/diff_test"
TOLERANCE = 1e-4

def lean_eval(cmd: str, *args) -> str:
    """Call Lean executable and return stdout."""
    inp = "|".join([cmd] + [str(a) for a in args])
    result = subprocess.run(
        [LEAN_EXE], input=inp, capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}"
    return result.stdout.strip()

def python_softmax(xs):
    import math
    exps = [math.exp(x) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def python_rmsnorm(xs, eps=1e-5):
    ms = sum(x * x for x in xs) / len(xs)
    denom = math.sqrt(ms + eps)
    return [x / denom for x in xs]

def python_layernorm(xs, eps=1e-5):
    m = sum(xs) / len(xs)
    variance = sum((x - m) ** 2 for x in xs) / len(xs)
    denom = math.sqrt(variance + eps)
    return [(x - m) / denom for x in xs]

def python_rope(xs, pos, base=10000):
    d = len(xs)
    result = list(xs)
    i = 0
    while i + 1 < d:
        freq = 1.0 / (base ** (i / d))
        theta = pos * freq
        c = math.cos(theta)
        s = math.sin(theta)
        x, y = xs[i], xs[i + 1]
        result[i] = c * x - s * y
        result[i + 1] = s * x + c * y
        i += 2
    return result

def python_active_fraction(variant, n_experts, top_k, n_shared):
    if variant == "dense":
        return 1.0
    elif variant == "switch":
        return 1.0 / n_experts
    elif variant == "moe":
        return top_k / n_experts
    elif variant == "deepseek":
        return (top_k + n_shared) / (n_experts + n_shared)
    return 0.0

def python_grad_magnitude(variant, depth, n_layers, w=0.8, streams=1):
    from_output = n_layers - depth
    if variant == "plain":
        return w ** from_output
    elif variant == "rc":
        return 1 + from_output * 0.1
    elif variant == "hc":
        return 1 + from_output * 0.1 * streams
    elif variant == "mhc":
        return 1 + from_output * 0.12 * streams
    elif variant == "attnres":
        return 1 + from_output * 0.08
    return 0.0

def compare_floats(a, b, tol=TOLERANCE):
    return abs(a - b) <= tol

def compare_lists(a, b, tol=TOLERANCE):
    if len(a) != len(b):
        return False
    return all(compare_floats(x, y, tol) for x, y in zip(a, b))

def test_softmax(iteration):
    n = random.randint(2, 8)
    xs = [random.uniform(-5, 5) for _ in range(n)]
    xs_str = ",".join(f"{x:.6f}" for x in xs)

    lean_out = lean_eval("softmax", xs_str)
    if not lean_out.startswith("SOFTMAX|"):
        return False, f"Lean error: {lean_out}"
    lean_vals = [float(v) for v in lean_out.split("|")[1].split(",")]

    py_vals = python_softmax(xs)

    if not compare_lists(lean_vals, py_vals):
        return False, f"softmax mismatch: xs={xs}\n  Lean={lean_vals}\n  Py  ={py_vals}"
    return True, ""

def test_rmsnorm(iteration):
    n = random.randint(2, 8)
    xs = [random.uniform(-3, 3) for _ in range(n)]
    eps = random.choice([1e-5, 1e-6, 1e-3])
    xs_str = ",".join(f"{x:.6f}" for x in xs)

    lean_out = lean_eval("rmsnorm", xs_str, f"{eps:.6f}")
    if not lean_out.startswith("RMSNORM|"):
        return False, f"Lean error: {lean_out}"
    lean_vals = [float(v) for v in lean_out.split("|")[1].split(",")]

    py_vals = python_rmsnorm(xs, eps)

    if not compare_lists(lean_vals, py_vals):
        return False, f"rmsnorm mismatch: xs={xs}, eps={eps}\n  Lean={lean_vals}\n  Py  ={py_vals}"
    return True, ""

def test_layernorm(iteration):
    n = random.randint(2, 8)
    xs = [random.uniform(-3, 3) for _ in range(n)]
    eps = random.choice([1e-5, 1e-6, 1e-3])
    xs_str = ",".join(f"{x:.6f}" for x in xs)

    lean_out = lean_eval("layernorm", xs_str, f"{eps:.6f}")
    if not lean_out.startswith("LAYERNORM|"):
        return False, f"Lean error: {lean_out}"
    lean_vals = [float(v) for v in lean_out.split("|")[1].split(",")]

    py_vals = python_layernorm(xs, eps)

    if not compare_lists(lean_vals, py_vals):
        return False, f"layernorm mismatch: xs={xs}, eps={eps}\n  Lean={lean_vals}\n  Py  ={py_vals}"
    return True, ""

def test_rope(iteration):
    n = random.randint(2, 8) * 2  # even length for pairs
    xs = [random.uniform(-2, 2) for _ in range(n)]
    pos = random.uniform(0, 100)
    base = random.choice([10000.0, 500.0, 1000000.0])
    xs_str = ",".join(f"{x:.6f}" for x in xs)

    lean_out = lean_eval("rope", xs_str, f"{pos:.6f}", f"{base:.1f}")
    if not lean_out.startswith("ROPE|"):
        return False, f"Lean error: {lean_out}"
    lean_vals = [float(v) for v in lean_out.split("|")[1].split(",")]

    py_vals = python_rope(xs, pos, base)

    if not compare_lists(lean_vals, py_vals):
        return False, f"rope mismatch: pos={pos}, base={base}\n  Lean={lean_vals}\n  Py  ={py_vals}"
    return True, ""

def test_active_fraction(iteration):
    variant = random.choice(["dense", "switch", "moe", "deepseek"])
    n_experts = random.randint(2, 16)
    top_k = random.randint(1, min(n_experts, 4))
    n_shared = random.randint(0, 4)

    lean_out = lean_eval("activeFraction", variant, n_experts, top_k, n_shared)
    if not lean_out.startswith("ACTIVEFRACTION|"):
        return False, f"Lean error: {lean_out}"
    lean_val = float(lean_out.split("|")[1])

    py_val = python_active_fraction(variant, n_experts, top_k, n_shared)

    if not compare_floats(lean_val, py_val):
        return False, f"activeFraction mismatch: {variant} nE={n_experts} k={top_k} s={n_shared}\n  Lean={lean_val}\n  Py  ={py_val}"
    return True, ""

def test_grad_magnitude(iteration):
    variant = random.choice(["plain", "rc", "hc", "mhc", "attnres"])
    n_layers = random.randint(5, 50)
    depth = random.randint(0, n_layers)
    w = random.uniform(0.1, 0.99)
    streams = random.randint(1, 4)

    lean_out = lean_eval("gradMagnitude", variant, depth, n_layers, f"{w:.6f}", streams)
    if not lean_out.startswith("GRADMAG|"):
        return False, f"Lean error: {lean_out}"
    lean_val = float(lean_out.split("|")[1])

    py_val = python_grad_magnitude(variant, depth, n_layers, w, streams)

    if not compare_floats(lean_val, py_val):
        return False, f"gradMagnitude mismatch: {variant} d={depth} nL={n_layers} w={w:.4f} s={streams}\n  Lean={lean_val}\n  Py  ={py_val}"
    return True, ""

TESTS = [
    ("softmax", test_softmax),
    ("rmsnorm", test_rmsnorm),
    ("layernorm", test_layernorm),
    ("rope", test_rope),
    ("activeFraction", test_active_fraction),
    ("gradMagnitude", test_grad_magnitude),
]

def main():
    parser = argparse.ArgumentParser(description="Differential testing: Lean vs Python")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    total_pass = 0
    total_fail = 0

    for name, test_fn in TESTS:
        passed = 0
        failed = 0
        for i in range(args.iterations):
            ok, msg = test_fn(i)
            if ok:
                passed += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  FAIL [{name} #{i}]: {msg}")
        total_pass += passed
        total_fail += failed
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {name:20s} {passed:4d}/{args.iterations}  [{status}]")

    print(f"\n  Total: {total_pass}/{total_pass + total_fail} passed")
    if total_fail == 0:
        print("  All tests passed!")
    else:
        print(f"  {total_fail} failures")
        sys.exit(1)

if __name__ == "__main__":
    main()
