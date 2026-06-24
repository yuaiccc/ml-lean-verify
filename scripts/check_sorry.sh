#!/usr/bin/env bash
# ============================================================
# check_sorry.sh — 扫描所有 .lean 文件中的 sorry
#
# 功能：
#   - 递归扫描项目中所有 .lean 文件
#   - 如果发现 sorry，打印文件名和行号，退出码 1
#   - 如果没有 sorry，打印 "No sorry found"，退出码 0
#
# 用法：
#   ./scripts/check_sorry.sh [目录路径]
#
#   默认扫描项目根目录（脚本所在目录的上一级）
# ============================================================

set -euo pipefail

# ── 颜色定义 ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# ── 确定项目根目录 ───────────────────────────────────────────
# 脚本位于 scripts/ 目录下，项目根目录为上一级
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 允许通过参数指定扫描目录，默认为项目根目录
SCAN_DIR="${1:-${PROJECT_ROOT}}"

echo -e "${YELLOW}扫描目录: ${SCAN_DIR}${NC}"
echo -e "${YELLOW}扫描所有 .lean 文件中的 sorry ...${NC}"
echo ""

# ── 扫描 sorry ───────────────────────────────────────────────
# 使用 grep 递归搜索 .lean 文件中的 sorry
# -r: 递归搜索
# -n: 显示行号
# -I: 忽略二进制文件
# --include="*.lean": 只搜索 .lean 文件
SORRY_FOUND=0
SORRY_OUTPUT=""

# 排除 .lake 目录（依赖库，不检查）
SORRY_OUTPUT=$(grep -rn --include="*.lean" "sorry" "${SCAN_DIR}" \
    --exclude-dir=".lake" \
    --exclude-dir=".git" 2>/dev/null) || true

if [ -n "${SORRY_OUTPUT}" ]; then
    SORRY_FOUND=1
fi

# ── 输出结果 ─────────────────────────────────────────────────
if [ "${SORRY_FOUND}" -eq 1 ]; then
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}发现 sorry！以下文件包含 sorry：${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    # 逐行打印，高亮显示
    while IFS= read -r line; do
        echo -e "  ${RED}${line}${NC}"
    done <<< "${SORRY_OUTPUT}"
    echo ""
    # 统计总数
    SORRY_COUNT=$(echo "${SORRY_OUTPUT}" | wc -l | tr -d ' ')
    echo -e "${RED}共发现 ${SORRY_COUNT} 处 sorry${NC}"
    echo -e "${RED}请移除所有 sorry，确保定理完整证明。${NC}"
    exit 1
else
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}No sorry found${NC}"
    echo -e "${GREEN}所有 .lean 文件均未使用 sorry，定理完整。${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi
