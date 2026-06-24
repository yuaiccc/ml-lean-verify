#!/usr/bin/env bash
# ============================================================
# verify_all.sh — 本地统一验证脚本
#
# 一键运行所有验证步骤，彩色输出显示每步状态，最终汇总报告。
#
# 验证步骤：
#   a) Lean 4 构建 + sorry 检查
#   b) 差分测试（Lean vs Python）
#   c) SMT 鲁棒性验证（如果 Python 依赖可用）
#   d) PGD 对抗攻击
#   e) 汇总报告
#
# 退出码：
#   0 = 全部通过
#   1 = 有失败
#
# 用法：
#   ./scripts/verify_all.sh [选项]
#
# 选项：
#   --skip-smt       跳过 SMT 验证步骤
#   --skip-pgd       跳过 PGD 攻击步骤
#   --iterations N   差分测试迭代次数（默认 50）
#   --help           显示帮助信息
# ============================================================

set -uo pipefail

# ── 颜色定义 ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # 无颜色

# ── 确定项目根目录 ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# ── 默认参数 ─────────────────────────────────────────────────
SKIP_SMT=false
SKIP_PGD=false
ITERATIONS=50

# ── 解析命令行参数 ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-smt)
            SKIP_SMT=true
            shift
            ;;
        --skip-pgd)
            SKIP_PGD=true
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: ./scripts/verify_all.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-smt       跳过 SMT 鲁棒性验证步骤"
            echo "  --skip-pgd       跳过 PGD 对抗攻击步骤"
            echo "  --iterations N   差分测试迭代次数（默认 50）"
            echo "  --help           显示此帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# ── 结果记录数组 ─────────────────────────────────────────────
declare -a STEP_NAMES=()
declare -a STEP_STATUS=()
declare -a STEP_DETAILS=()

# ── 辅助函数 ─────────────────────────────────────────────────

# 打印步骤标题
print_step_header() {
    local step_num="$1"
    local step_name="$2"
    echo ""
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}${BOLD}步骤 ${step_num}: ${step_name}${NC}"
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# 记录步骤结果
record_result() {
    local name="$1"
    local status="$2"  # PASS / FAIL / SKIP
    local detail="$3"
    STEP_NAMES+=("${name}")
    STEP_STATUS+=("${status}")
    STEP_DETAILS+=("${detail}")
}

# 打印步骤结果
print_step_result() {
    local status="$1"
    local detail="$2"
    case "${status}" in
        PASS)
            echo -e "${GREEN}${BOLD}[ 通过 ]${NC} ${detail}"
            ;;
        FAIL)
            echo -e "${RED}${BOLD}[ 失败 ]${NC} ${detail}"
            ;;
        SKIP)
            echo -e "${YELLOW}${BOLD}[ 跳过 ]${NC} ${detail}"
            ;;
    esac
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ── 打印总标题 ───────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          ml-lean-verify 统一验证脚本                      ║"
echo "║          Lean 4 形式化 + SMT 鲁棒性 + PGD 攻击            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "项目根目录: ${YELLOW}${PROJECT_ROOT}${NC}"
echo -e "差分测试迭代次数: ${YELLOW}${ITERATIONS}${NC}"
echo -e "当前时间: ${YELLOW}$(date '+%Y-%m-%d %H:%M:%S')${NC}"

# ============================================================
# 步骤 a: Lean 4 构建 + sorry 检查
# ============================================================
print_step_header "a" "Lean 4 构建 + sorry 检查"

LEAN_BUILD_OK=true
LEAN_BUILD_DETAIL=""

# 检查 lake 命令是否可用
if ! command_exists lake; then
    echo -e "${RED}错误: 未找到 lake 命令，请先安装 Lean 4 工具链 (elan)。${NC}"
    echo -e "${YELLOW}安装命令: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y${NC}"
    record_result "Lean 4 构建" "FAIL" "lake 命令不可用"
    LEAN_BUILD_OK=false
else
    # lake build
    echo -e "${CYAN}运行 lake build ...${NC}"
    if lake build 2>&1; then
        echo -e "${GREEN}lake build 成功${NC}"
        LEAN_BUILD_DETAIL="lake build 成功"
    else
        echo -e "${RED}lake build 失败${NC}"
        record_result "Lean 4 构建" "FAIL" "lake build 失败"
        LEAN_BUILD_OK=false
    fi
fi

# sorry 检查（仅在构建成功时）
if [ "${LEAN_BUILD_OK}" = true ]; then
    echo ""
    echo -e "${CYAN}运行 sorry 检查 ...${NC}"
    if bash "${SCRIPT_DIR}/check_sorry.sh" "${PROJECT_ROOT}"; then
        record_result "Lean 4 构建 + sorry 检查" "PASS" "${LEAN_BUILD_DETAIL}, 零 sorry"
    else
        record_result "Lean 4 构建 + sorry 检查" "FAIL" "发现 sorry"
        LEAN_BUILD_OK=false
    fi
fi

# ============================================================
# 步骤 b: 差分测试
# ============================================================
print_step_header "b" "差分测试（Lean vs Python）"

if [ "${LEAN_BUILD_OK}" = true ]; then
    # 构建 diff_test 可执行端
    echo -e "${CYAN}构建 diff_test 可执行端 ...${NC}"
    if lake build diff_test 2>&1; then
        echo -e "${GREEN}diff_test 构建成功${NC}"
        # 运行差分测试
        echo -e "${CYAN}运行差分测试（${ITERATIONS} 次迭代）...${NC}"
        if python3 diff_test.py --iterations "${ITERATIONS}" 2>&1; then
            record_result "差分测试" "PASS" "全部通过（${ITERATIONS} 次迭代）"
        else
            record_result "差分测试" "FAIL" "差分测试存在失败"
        fi
    else
        echo -e "${RED}diff_test 构建失败${NC}"
        record_result "差分测试" "FAIL" "diff_test 构建失败"
    fi
else
    echo -e "${YELLOW}跳过差分测试（Lean 构建未通过）${NC}"
    record_result "差分测试" "SKIP" "依赖 Lean 构建"
fi

# ============================================================
# 步骤 c: SMT 鲁棒性验证
# ============================================================
print_step_header "c" "SMT 鲁棒性验证（Z3）"

if [ "${SKIP_SMT}" = true ]; then
    echo -e "${YELLOW}已跳过 SMT 验证（--skip-smt）${NC}"
    record_result "SMT 鲁棒性验证" "SKIP" "用户指定跳过"
else
    # 检查 Python 依赖是否可用
    PYTHON_DEPS_OK=true
    MISSING_DEPS=""

    for dep in torch torchvision z3 scipy matplotlib numpy; do
        if ! python3 -c "import ${dep}" 2>/dev/null; then
            PYTHON_DEPS_OK=false
            MISSING_DEPS="${MISSING_DEPS} ${dep}"
        fi
    done

    if [ "${PYTHON_DEPS_OK}" = true ]; then
        # 训练模型
        echo -e "${CYAN}训练 MNIST MLP 模型 ...${NC}"
        if python3 smt_verify/train_model.py 2>&1; then
            echo -e "${GREEN}模型训练成功${NC}"

            # SMT 验证
            echo -e "${CYAN}运行 Z3 SMT 鲁棒性验证 ...${NC}"
            if python3 smt_verify/verify_robustness.py 2>&1; then
                record_result "SMT 鲁棒性验证" "PASS" "Z3 验证完成"
            else
                record_result "SMT 鲁棒性验证" "FAIL" "Z3 验证失败"
            fi
        else
            echo -e "${RED}模型训练失败${NC}"
            record_result "SMT 鲁棒性验证" "FAIL" "模型训练失败"
        fi
    else
        echo -e "${YELLOW}Python 依赖不可用，跳过 SMT 验证${NC}"
        echo -e "${YELLOW}缺失的依赖:${MISSING_DEPS}${NC}"
        echo -e "${YELLOW}安装命令: pip install z3-solver torch torchvision matplotlib scipy numpy${NC}"
        record_result "SMT 鲁棒性验证" "SKIP" "Python 依赖不可用"
    fi
fi

# ============================================================
# 步骤 d: PGD 对抗攻击
# ============================================================
print_step_header "d" "PGD 对抗攻击"

if [ "${SKIP_PGD}" = true ]; then
    echo -e "${YELLOW}已跳过 PGD 攻击（--skip-pgd）${NC}"
    record_result "PGD 对抗攻击" "SKIP" "用户指定跳过"
else
    # PGD 依赖 SMT 验证步骤训练的模型
    # 检查模型文件是否存在
    MODEL_PATH="${PROJECT_ROOT}/smt_verify/mnist_tiny.pt"
    SAMPLES_PATH="${PROJECT_ROOT}/smt_verify/test_samples.npy"

    # 检查 Python 依赖
    PYTHON_DEPS_OK=true
    for dep in torch numpy; do
        if ! python3 -c "import ${dep}" 2>/dev/null; then
            PYTHON_DEPS_OK=false
        fi
    done

    if [ "${PYTHON_DEPS_OK}" = true ] && [ -f "${MODEL_PATH}" ] && [ -f "${SAMPLES_PATH}" ]; then
        echo -e "${CYAN}运行 PGD 对抗攻击 ...${NC}"
        if python3 smt_verify/pgd_attack.py 2>&1; then
            record_result "PGD 对抗攻击" "PASS" "PGD 攻击完成"
        else
            record_result "PGD 对抗攻击" "FAIL" "PGD 攻击失败"
        fi
    else
        if [ ! -f "${MODEL_PATH}" ] || [ ! -f "${SAMPLES_PATH}" ]; then
            echo -e "${YELLOW}模型文件或测试样本不存在，跳过 PGD 攻击${NC}"
            echo -e "${YELLOW}请先运行 SMT 验证步骤以训练模型${NC}"
            record_result "PGD 对抗攻击" "SKIP" "模型文件不存在"
        else
            echo -e "${YELLOW}Python 依赖不可用，跳过 PGD 攻击${NC}"
            echo -e "${YELLOW}安装命令: pip install torch numpy${NC}"
            record_result "PGD 对抗攻击" "SKIP" "Python 依赖不可用"
        fi
    fi
fi

# ============================================================
# 步骤 e: 汇总报告
# ============================================================
print_step_header "e" "汇总报告"

echo -e "${CYAN}${BOLD}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}${BOLD}│                    验证结果汇总                          │${NC}"
echo -e "${CYAN}${BOLD}├──────────────────────────────────────────────────────────┤${NC}"

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
OVERALL_SUCCESS=true

for i in "${!STEP_NAMES[@]}"; do
    name="${STEP_NAMES[$i]}"
    status="${STEP_STATUS[$i]}"
    detail="${STEP_DETAILS[$i]}"

    case "${status}" in
        PASS)
            symbol="${GREEN}${BOLD}[PASS]${NC}"
            TOTAL_PASS=$((TOTAL_PASS + 1))
            ;;
        FAIL)
            symbol="${RED}${BOLD}[FAIL]${NC}"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            OVERALL_SUCCESS=false
            ;;
        SKIP)
            symbol="${YELLOW}${BOLD}[SKIP]${NC}"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            ;;
    esac

    # 格式化输出，对齐列宽
    printf "${CYAN}${BOLD}│${NC} %-38s %b  ${CYAN}${BOLD}│${NC}\n" "${name}" "${symbol}"
done

echo -e "${CYAN}${BOLD}├──────────────────────────────────────────────────────────┤${NC}"
echo -e "${CYAN}${BOLD}│${NC} 通过: ${GREEN}${TOTAL_PASS}${NC}  失败: ${RED}${TOTAL_FAIL}${NC}  跳过: ${YELLOW}${TOTAL_SKIP}${NC}                        ${CYAN}${BOLD}│${NC}"
echo -e "${CYAN}${BOLD}└──────────────────────────────────────────────────────────┘${NC}"

echo ""

# 打印详细结果
echo -e "${BOLD}详细结果:${NC}"
for i in "${!STEP_NAMES[@]}"; do
    name="${STEP_NAMES[$i]}"
    status="${STEP_STATUS[$i]}"
    detail="${STEP_DETAILS[$i]}"
    print_step_result "${status}" "${name} — ${detail}"
done

echo ""

# ── 最终退出码 ───────────────────────────────────────────────
if [ "${OVERALL_SUCCESS}" = true ]; then
    echo -e "${GREEN}${BOLD}========================================${NC}"
    echo -e "${GREEN}${BOLD}  全部验证通过！${NC}"
    echo -e "${GREEN}${BOLD}========================================${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}========================================${NC}"
    echo -e "${RED}${BOLD}  验证失败！请检查上述失败步骤。${NC}"
    echo -e "${RED}${BOLD}========================================${NC}"
    exit 1
fi
