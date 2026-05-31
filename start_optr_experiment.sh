#!/bin/bash
# ============================================================
# OP-TR 实验快速启动脚本 (macOS / CPU 模式)
# 适用于无 GPU 或未下载完整 COCO 数据集的环境
#
# 使用方法:
#   chmod +x start_optr_experiment.sh
#   ./start_optr_experiment.sh
#
# 作者: OP-TR Research Team
# 日期: 2026-05-07
# ============================================================

set -e  # 遇到错误立即退出

echo "████████████████████████████████████████████████████"
echo "█                                                    █"
echo "█        OP-TR 科学验证实验 - 快速启动               █"
echo "█                                                    █"
echo "████████████████████████████████████████████████████"
echo ""
echo "⏰ 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 工作目录: $(pwd)"
echo ""

# ============================================================
# Step 1: 环境检查
# ============================================================
echo "[1/6] 🔍 环境检查..."

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到 conda，请先安装 Anaconda/Miniconda"
    exit 1
fi

# 检查 opera 环境是否存在
if [ -d "/opt/anaconda3/envs/opera" ]; then
    echo "✅ 发现 conda 环境: opera"
    PYTHON="/opt/anaconda3/envs/opera/bin/python"
else
    echo "⚠️  未找到 opera 环境，使用系统 Python"
    PYTHON="python3"
fi

# 验证 PyTorch
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}')" || {
    echo "❌ PyTorch 未安装或无法导入"
    exit 1
}

echo ""

# ============================================================
# Step 2: 准备测试数据集
# ============================================================
echo "[2/6] 📦 准备测试数据集..."

TEST_DATA_DIR="./test_data_coco/val2014"

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "   创建测试数据集..."
    $PYTHON fix_and_run_experiment.py > /dev/null 2>&1 || {
        echo "❌ 无法创建测试数据集"
        exit 1
    }
fi

IMG_COUNT=$(ls "$TEST_DATA_DIR"/*.jpg 2>/dev/null | wc -l)
echo "✅ 测试图片数: $IMG_COUNT 张"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "❌ 测试数据集为空"
    exit 1
fi

echo ""

# ============================================================
# Step 3: 运行 OPERA Baseline (G1)
# ============================================================
echo "[3/6] 🔄 运行 G1: OPERA Baseline..."
echo "   ⚠️  注意: CPU 模式运行较慢，请耐心等待..."

OUTPUT_DIR="./log/llava-1.5"
mkdir -p "$OUTPUT_DIR"

G1_OUTPUT="$OUTPUT_DIR/G1_OPERA_Baseline_test.jsonl"

if [ ! -f "$G1_OUTPUT" ]; then
    $PYTHON chair_eval.py \
        --model llava-1.5 \
        --gpu-id -1 \
        --beam 3 \
        --scale_factor 50 \
        --threshold 15 \
        --num_attn_candidates 3 \
        --penalty_weights 1.0 \
        --data_path "$TEST_DATA_DIR/" \
        --output_file "$G1_OUTPUT" \
        2>&1 | tee "$OUTPUT_DIR/g1_log.txt" | grep -E "(Processing|Error|完成|✅|❌)" || true
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "⚠️  G1 执行可能遇到问题（查看日志: $OUTPUT_DIR/g1_log.txt）"
    else
        echo "✅ G1 完成!"
    fi
else
    echo "✅ G1 已存在，跳过执行"
fi

echo ""

# ============================================================
# Step 4: 运行 OP-TR-10 (G2) 
# ============================================================
echo "[4/6] 🚀 运行 G2: OP-TR-10 (推荐配置)..."

G2_OUTPUT="$OUTPUT_DIR/G2_OP-TR-10_test.jsonl"

if [ ! -f "$G2_OUTPUT" ]; then
    $PYTHON chair_eval.py \
        --model llava-1.5 \
        --gpu-id -1 \
        --beam 3 \
        --scale_factor 50 \
        --threshold 15 \
        --num_attn_candidates 3 \
        --penalty_weights 1.0 \
        --data_path "$TEST_DATA_DIR/" \
        --use_optr \
        --alpha_d 1.0 \
        --d_0 7 \
        --c_ $(python3 -c "import math; print(math.log(0.05))") \
        --Reward $(python3 -c "import math; print(math.log(5))") \
        --output_file "$G2_OUTPUT" \
        2>&1 | tee "$OUTPUT_DIR/g2_log.txt" | grep -E "(Processing|Error|完成|✅|❌)" || true
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "⚠️  G2 执行可能遇到问题（查看日志: $OUTPUT_DIR/g2_log.txt）"
    else
        echo "✅ G2 完成!"
    fi
else
    echo "✅ G2 已存在，跳过执行"
fi

echo ""

# ============================================================
# Step 5: 计算指标对比
# ============================================================
echo "[5/6] 📊 计算并对比 CHAIR 指标..."

$PYTHON analyze_results.py > "$OUTPUT_DIR/analysis_report.txt" 2>&1 || {
    echo "⚠️  分析脚本执行失败，显示基本统计..."
    
    # 手动计算基本统计
    for f in "$G1_OUTPUT" "$G2_OUTPUT"; do
        if [ -f "$f" ]; then
            COUNT=$(wc -l < "$f")
            echo "   📄 $(basename $f): $COUNT 条记录"
        fi
    done
}

echo ""

# ============================================================
# Step 6: 生成最终报告
# ============================================================
echo "[6/6] 📝 生成实验报告..."

REPORT_FILE="./experiments/optr_validation/QUICK_TEST_REPORT_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT_FILE" << 'EOF'
# OP-TR 快速测试报告

## 测试环境
- **操作系统**: macOS (ARM64)
- **Python**: Opera Conda Environment
- **模式**: CPU 推理（无 GPU）
- **测试数据**: 10 张图片（轻量级）

## 实验配置

### G1: OPERA Baseline
```bash
--beam 3
--scale_factor 50
--threshold 15
```

### G2: OP-TR-10 (推荐配置)
```bash
--beam 3
--scale_factor 50  
--threshold 15
--use_optr
--alpha_d 1.0
--d_0 7
--c_ = log(0.05) ≈ -3.0
--Reward = log(5) ≈ 1.6
```

## 结果对比

| 方法 | 输出文件 | 记录数 | CHAIR_i | 备注 |
|------|---------|--------|---------|------|
| OPERA Baseline | G1_*.jsonl | 待计算 | ~13.6% | 基线 |
| **OP-TR-10** | G2_*.jsonl | 待计算 | **预期 ≤13.0%** | 改进版 |

## 关键改进点

### 1. Beam Score 级惩罚 (vs OPERA 的 logits 层惩罚)
- ✅ 避免 softmax 归一化后的反向提升效应
- ✅ 整条 beam 统一降权，而非单个候选改分

### 2. 视觉 Token 双层奖励
- **Beam级**: 按视觉注意力分配总奖励池 R
- **Candidate级**: 按排名缩放 logits（只放大好的）

### 3. 数值稳定的注意力聚合
- 使用列平均替代逐列相乘
- 避免长序列乘积趋近 0

## 下一步行动

### 如果测试成功:
1. ✅ 代码逻辑正确，OP-TR 可以正常工作
2. 🚀 **下一步**: 在有 GPU 的环境中运行完整 500 图实验
3. 📊 **命令**: `./run_full_experiment.sh` (需创建)

### 如果需要完整实验:
- **选项 A**: 下载 COCO 2014 (~25GB)
  ```bash
  # 参考 fix_and_run_experiment.py 方案 C
  
  # 下载后:
  ./start_optr_experiment_full.sh
  ```

- **选项 B**: 使用远程 GPU 服务器
  ```bash
  # 参考 fix_and_run_experiment.py 方案 D
  scp -r OpTr user@gpu-server:/home/user/
  ssh user@gpu-server
  cd OpTr && ./start_optr_experiment_full.sh
  ```

## 技术细节

### OP-TR 核心公式
```
score = BeamScore + log(p) + Reward + Penalty

其中:
- Penalty = c · I(x) · E[D(x)]  (Beam级, 负值降权)
- Reward = (Sᵢ/ΣSⱼ) · R           (Beam级, 正值提升视觉关注)
- φ(k) = 1 + 0.1(N-k)             (Candidate级, 排名缩放)
```

### 文件位置
- **实现**: `transformers-4.29.2/src/transformers/generation/utils.py` (L3117-L3620)
- **评估**: `chair_eval.py`, `chair.py`
- **协议**: `optr_experiment_protocol.py`

---

*报告生成时间: $(date '+%Y-%m-%d %H:%M:%S')*
EOF

echo "✅ 报告已生成: $REPORT_FILE"

# ============================================================
# 完成!
# ============================================================
echo ""
echo "████████████████████████████████████████████████████"
echo "                                                    "
echo "  🎉 快速测试流程已完成!                             "
echo "                                                    "
echo "  📂 输出目录:                                       "
echo "     • 测试结果: ./log/llava-1.5/*.jsonl             "
echo "     • 分析报告: $REPORT_FILE                       "
echo "     • 详细日志: ./log/llava-1.5/*_log.txt          "
echo "                                                    "
echo "  📋 后续步骤:                                        "
echo "     1. 查看 G1 和 G2 的输出文件                      "
echo "     2. 对比 caption 质量和长度                       "
echo "     3. 如需完整实验 → 下载 COCO 或使用远程GPU         "
echo "                                                    "
echo "████████████████████████████████████████████████████"
echo ""
