# OP-TR 科学验证实验方案 - 快速启动指南

## 📋 方案概述

本实验方案旨在**全面、严谨地验证 OP-TR (Over-trust Penalty with Trust Reward)** 改进措施在降低视觉语言模型幻觉方面的有效性。

### 核心目标
- **主要目标**: 验证 OP-TR 能将 CHAIR_i 从 OPERA 的 13.6% 降低到 ≤13.0% (降幅 ≥5%)
- **次要目标**: 通过 14 组消融实验量化各组件的独立贡献

### 实验规模
- **总实验组**: 15 组 (G0-G14)
- **样本量**: 500 张 COCO 2014 图像/组
- **预计耗时**: 35-47 小时 (2-3 个工作日)

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd /Users/seyonmacbook/Desktop/电子书/paper复现/OpTr

# 安装依赖
pip install -r requirements.txt

# 确认 GPU 可用
nvidia-smi
```

### 2. 运行完整实验套件

```bash
# 基础运行（所有 15 组）
python optr_experiment_protocol.py --mode run --gpu-id 0

# 仅运行核心实验（G0-G7，节省时间）
python optr_experiment_protocol.py --mode run --groups G0 G1 G2 G3 G4 G5 G6 G7 --gpu-id 0

# 仅运行主实验组（最快验证）
python optr_experiment_protocol.py --mode run --groups G1 G2 G3 --gpu-id 0
```

### 3. 分析结果

```bash
# 统计分析
python optr_experiment_protocol.py --mode analyze

# 生成报告
python optr_experiment_protocol.py --mode report
```

---

## 📊 实验组设计详解

### 第一层: 基线对照 (G0-G1)

| 组ID | 方法 | 目的 | 预期 CHAIR_i |
|------|------|------|-------------|
| **G0** | Standard Beam Search | 无反 hallucination 机制 | ~18-20% |
| **G1** | OPERA Baseline | 当前最佳方法（logits 层惩罚） | **13.6%** (基线) |

### 第二层: 主实验组 (G2-G3) ⭐ 推荐

| 组ID | 配置名称 | 关键参数 | 设计理念 | 目标 CHAIR_i |
|------|---------|---------|---------|-------------|
| **G2** | **OP-TR-10** | α=1.0, d₀=7, c=log(0.05), R=log(5) | 平衡惩罚与奖励 | **≤13.0%** ✅ |
| **G3** | OP-TR-12 | α=0.8, d₀=6, c=log(0.005), R=log(15) | 更激进干预 | **≤12.8%** ✅ |

**推荐优先完成 G2，这是论文的主要结果配置**

### 第三层: 核心消融实验 (G4-G7)

| 组ID | 消融类型 | 修改内容 | 验证假设 |
|------|---------|---------|---------|
| **G4** | 仅惩罚 | Reward=0 | H3: 惩罚的独立贡献 |
| **G5** | 仅奖励 | c_=0 | H3: 奖励的独立贡献 |
| **G6** | 仅 Beam 奖励 | 禁用 Candidate 级 φ | 候选级奖励的作用 |
| **G7** | 无距离惩罚 | d₀=1000 | 距离因子 D(x) 的必要性 |

**预期结果**: 
- G4, G5 的 CHAIR_i 应 ≈13.5%（比完整 OP-TR 高 0.5%）
- 这证明**惩罚和奖励都是必要的**，且存在协同效应

### 第四层: 敏感性分析 (G8-G11)

| 组ID | 变化参数 | 测试值 | 目的 |
|------|---------|-------|------|
| **G8** | alpha_d | 0.5 (更平滑) | 距离衰减敏感性 |
| **G9** | alpha_d | 1.5 (更陡峭) | 距离衰减敏感性 |
| **G10** | d_0 | 5 (更早触发) | 距离阈值敏感性 |
| **G11** | d_0 | 9 (更晚触发) | 距离阈值敏感性 |

### 第五层: 强度分析 (G12-G14)

| 组ID | 特征 | 应用场景 |
|------|------|---------|
| **G12** | 强惩罚 (c=log 0.001) | 幻觉严重时的激进策略 |
| **G13** | 强奖励 (R=log 20) | 视觉关注度不足时 |
| **G14** | 极端干预 | **鲁棒性测试**（可能退化）|

---

## 🔬 研究假设与检验方法

### 主要假设

```
H1: OP-TR 的 CHAIR_i 显著低于 OPERA
   → 检验方法: Paired Wilcoxon signed-rank test
   → 显著性水平: p < 0.05
   → 效应量: Cohen's d > 0.5 (中等效应)
   
H2: Beam Score 级惩罚 > logits 层惩罚
   → 对比: G2/G3 vs G1 的惩罚机制差异
   
H3: 惩罚和奖励均有独立贡献
   → 对比: G2 vs G4 (证明奖励重要)
          G2 vs G5 (证明惩罚重要)
   → 预期: G4, G5 的 CHAIR_i 比 G2 高 ≥0.5%
   
H4: 存在协同效应 (1+1>2)
   → 检验: G2 < min(G4, G5) - 误差范围
```

### 统计检验流程

```python
# 伪代码：主要统计检验
from scipy import stats
import numpy as np

# 1. 主要比较 (H1)
opera_chair_i = results['G1']['per_image_chair_i']  # 500 values
optr10_chair_i = results['G2']['per_image_chair_i']

statistic, p_value = stats.wilcoxon(opera_chair_i, optr10_chair_i)
cohens_d = compute_cohens_d(opera_chair_i, optr10_chair_i)

print(f"OP-TR-10 vs OPERA:")
print(f"  Wilcoxon p-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  Mean difference: {np.mean(optr10_chair_i) - np.mean(opera_chair_i):.4f}")

# 2. 消融实验 ANOVA
groups_data = [
    results['G1']['per_image_chair_i'],  # OPERA
    results['G2']['per_image_chair_i'],  # Full OP-TR
    results['G4']['per_image_chair_i'],  # Penalty only
    results['G5']['per_image_chair_i'],  # Reward only
]
f_stat, p_anova = stats.f_oneway(*groups_data)
```

---

## 📈 评价指标体系

### 主要指标 (Primary)

| 指标 | 全称 | 定义 | 目标方向 | 成功标准 |
|------|------|------|---------|---------|
| **CHAIR_i** | Instance-level Hallucination Rate | 幻觉实例数 / 总对象提及数 | ↓ 越低越好 | **≤13.0%** (vs 13.6%) |

### 次要指标 (Secondary)

| 指标 | 定义 | 约束条件 | 可接受范围 |
|------|------|---------|-----------|
| CHAIR_s | 包含幻觉的句子比例 | 应下降 | - |
| Recall | 正确识别的对象覆盖率 | **不应显著下降** | 降幅 ≤2% |
| Avg_Len | 平均 caption 长度 | 不应过度抑制 | ±10% of baseline |
| Inference_Time | 单张图推理时间 | 实用性考量 | ≤2x standard beam |

### OP-TR 特有指标 (Novel)

| 指标 | 用途 | 预期变化 |
|------|------|---------|
| Visual_Attention_Score | 量化视觉奖励效果 | ↑ 增加 |
| Penalty_Magnitude | 监控惩罚强度分布 | 与 CHAIR_i 负相关 |
| Reward_Distribution_Entropy | 检测奖励集中现象 | 保持稳定 |
| Candidate_Ranking_Correlation | 验证候选级奖励有效性 | ρ > 0.3 |

---

## ⏱️ 实验周期安排

### Phase 1: 准备工作 (2-3 小时)
- [ ] 验证数据集完整性 (500 images + annotations)
- [ ] 加载 LLaVA-1.5-7B 模型并验证权重
- [ ] 运行 sanity check (5 images, Standard Beam Search)
- [ ] 确认 OPERA 基线 CHAIR_i ∈ [13.4, 13.8]
- [ ] 创建输出目录结构

### Phase 2: 基线实验 (4-6 小时) ⭐ 优先
- [ ] **G0**: Standard Beam Search (~2h)
- [ ] **G1**: OPERA Baseline (~2-3h)
- [ ] 质量控制: 验证可重复性 (run twice)

### Phase 3: 主实验 (4-6 小时) ⭐⭐ 最重要
- [ ] **G2**: OP-TR-10 推荐配置 (~2h)
- [ ] **G3**: OP-TR-12 激进配置 (~2h)
- [ ] 实时监控: 检查中间指标趋势

### Phase 4: 消融实验 (8-10 小时)
- [ ] **G4**: Penalty-only (~2h)
- [ ] **G5**: Reward-only (~2h)
- [ ] **G6**: Beam-reward-only (~2h)
- [ ] **G7**: No-distance-penalty (~2h)

### Phase 5: 敏感性分析 (12-15 小时) - 可选
- [ ] Alpha 敏感性 (G8-G9, ~4h)
- [ ] d0 敏感性 (G10-G11, ~4h)
- [ ] 强度分析 (G12-G13, ~4h)

### Phase 6: 鲁棒性检查 (2-3 小时)
- [ ] **G14**: Extreme case (~2h)
- [ ] 监控数值稳定性 (NaN/Inf)

### Phase 7: 分析与报告 (3-4 小时)
- [ ] 计算所有 CHAIR 指标
- [ ] 执行统计显著性检验
- [ ] 生成可视化图表
- [ ] 撰写实验报告

---

## 🎯 成功标准清单

### 必须达成 (Must-Have) ✅

- [ ] **H1 成立**: OP-TR-10 CHAIR_i ≤ 13.0 (相对 OPERA 下降 ≥5%)
- [ ] **统计显著**: Wilcoxon test p-value < 0.05
- [ ] **效应量达标**: Cohen's d > 0.5 (中等效应)
- [ ] **可重复性**: 同一配置两次运行 MD5 完全一致

### 应该达成 (Should-Have) 📊

- [ ] **消融验证**: G4, G5 的 CHAIR_i 比 G2 高 ≥0.5%
- [ ] **无副作用**: Recall 下降 <2%, Avg_Len 变化 <10%
- [ ] **实用性**: 推理时间 ≤2x standard beam search

### 可以有 (Nice-to-Have) 💡

- [ ] **协同效应**: G2 < min(G4, G5) - 0.2%
- [ ] **参数单调性**: alpha/d0 与 CHAIR_i 呈现合理单调关系
- [ ] **极端鲁棒**: G14 未出现严重退化 (CHAIR_i < 15%)

---

## 🔧 故障排查

### 常见问题

#### Q1: CUDA Out of Memory
```bash
# 解决方案: 减少 batch_size 或使用 gradient checkpointing
python optr_experiment_protocol.py --gpu-id 0  # 默认 batch_size=1
```

#### Q2: 某些配置产生 NaN/Inf
```python
# 可能原因: 惩罚/奖励过强导致数值溢出
# 解决方案: 
# - 检查 G14 (极端配置) 是否异常
# - 调整 c_ 或 Reward 到合理范围
# - 添加梯度裁剪: torch.nn.utils.clip_grad_norm_(...)
```

#### Q3: 与 OPERA 基线差距过大 (>1%)
```bash
# 可能原因:
# 1. 数据顺序不一致 → 固定 random seed
# 2. 模型版本不同 → 检查 checkpoint hash
# 3. 预处理差异 → 对比 image normalization 参数

# 验证命令:
md5sum log/chair_eval_results/llava-1.5/ours.jsonl
# 应与文献报告的结果一致
```

#### Q4: 实验时间超出预期
```bash
# 分批运行策略:
# Batch 1 (必须): --groups G0 G1 G2 G3     # ~10-12h
# Batch 2 (消融): --groups G4 G5 G6 G7     # ~8-10h  
# Batch 3 (敏感): --groups G8-G13           # ~12-15h
# Batch 4 (极端): --groups G14              # ~2-3h
```

---

## 📁 输出文件说明

运行完成后，将在 `experiments/optr_exp_{timestamp}/` 生成：

```
experiments/optr_exp_20260507_143022/
├── README.md                          # 实验说明
├── experiment_config.json             # 完整配置记录
├── experimental_groups.json           # 15组参数定义
├── final_report.json                  # 结果汇总
├── captions/                          # ★ 生成的 captions (JSONL)
│   ├── G0_Standard_Beam_Search_a0_d0_0_c0_R0.jsonl
│   ├── G1_OPERA_Baseline_a0_d0_0_c0_R0.jsonl
│   ├── G2_OP_TR_10_a1_0_d0_7_c_log0_05_R_log1_61.jsonl  # ← 重点!
│   └── ... (共15个文件)
├── metrics/                           # CHAIR 指标计算结果
│   ├── G0_metrics.json
│   ├── G2_metrics.json               # ← 重点! 应显示 CHAIR_i ≤ 0.130
│   └── ...
├── logs/                              # 执行日志 & 进度
│   └── progress.json                  # 实时进度追踪
├── intermediate/                      # 中间数据 (用于调试)
│   ├── attentions/                    # 注意力权重矩阵
│   └── scores/                        # beam scores, penalties, rewards
├── analysis/                          # 统计分析结果
│   ├── statistical_tests.json         # Wilcoxon, ANOVA 结果
│   └── effect_sizes.json              # Cohen's d 等
└── figures/                           # ★ 可视化图表
    ├── 01_main_comparison.png        # 主对比柱状图
    ├── 02_ablation_heatmap.png       # 消融实验热力图
    ├── 03_sensitivity_curves.png     # 参数敏感性曲线
    └── 04_penalty_reward_scatter.png # 惩罚-奖励散点图
```

---

## 📝 下一步行动

### 立即执行 (Today)

1. **运行最小验证集** (30分钟):
   ```bash
   python optr_experiment_protocol.py --mode run --groups G1 G2 --gpu-id 0
   ```
   验证 OP-TR-10 能否达到 CHAIR_i ≤ 13.0%

2. **检查初步结果**:
   ```bash
   cat experiments/optr_exp_*/metrics/G2_metrics.json | python -m json.tool
   ```

### 本周完成 (This Week)

3. **完成全部实验** (如果 G2 达标):
   ```bash
   python optr_experiment_protocol.py --mode run --gpu-id 0  # 全部15组
   ```

4. **生成完整报告**:
   ```bash
   python optr_experiment_protocol.py --mode report
   ```

### 论文撰写准备 (Next Week)

5. **提取关键图表**:
   - Figure 1: 主对比图 (G0-G3)
   - Table 1: 完整结果表 (G0-G14)
   - Figure 2: 消融实验 (G1-G7)
   - Appendix: 敏感性分析 (G8-G14)

6. **撰写实验部分**:
   - 4.1 Experimental Setup
   - 4.2 Main Results (H1 验证)
   - 4.3 Ablation Study (H3/H4 验证)
   - 4.4 Analysis (敏感性 + 鲁棒性)

---

## 📚 相关资源

- **OP-TR 实现**: `transformers-4.29.2/src/transformers/generation/utils.py` (第3117行起)
- **评估脚本**: `chair.py`, `chair_eval.py`
- **已有基线**: `log/chair_eval_results/llava-1.5/ours.jsonl`
- **实验协议**: `optr_experiment_protocol.py` (本文档)

---

## 💡 提示与建议

1. **先跑 G2**: 如果 OP-TR-10 未达目标，调整参数后再跑全套
2. **保存中间数据**: `intermediate/` 目录对 debugging 很有用
3. **监控 GPU 使用**: `watch -n 1 nvidia-smi` 防止 OOM
4. **版本控制**: 每次 major 修改后 git commit
5. **文档先行**: 先写好实验设计再跑代码，避免返工

---

**祝实验顺利！🎉**

如有问题，请参考:
- 实验方案完整版: `optr_experiment_protocol.py` (含详细注释)
- OP-TR 实现代码: `utils.py` 第3117-3620行
- CHAIR 评估工具: `chair.py`

Last Updated: 2026-05-07
Version: v1.0 (Scientific Validation Protocol)
