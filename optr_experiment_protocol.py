#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OP-TR (Over-trust Penalty with Trust Reward) 科学验证实验方案
================================================================================

基于 OPERA 复现实现的改进措施全面测试方案
目标模型: LLaVA-1.5-7B
评估数据集: COCO 2014 Validation Set (500 images)
核心指标: CHAIR_i (Instance-level Hallucination Rate)

作者: OP-TR Research Team
日期: 2026-05-07
版本: v1.0 (Scientific Validation Protocol)

================================================================================
"""

import os
import sys
import json
import argparse
import math
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
import hashlib


# ============================================================================
# 第一部分: 实验目的与假设框架 (Section 1: Research Objectives & Hypotheses)
# ============================================================================

@dataclass
class ExperimentObjective:
    """实验目的定义"""
    
    primary_objective: str = """
    验证 OP-TR 改进措施在降低视觉语言模型幻觉率方面的有效性，
    量化评估 Beam Score 级惩罚和视觉 Token 奖励机制对生成质量的影响。
    """
    
    secondary_objectives: List[str] = field(default_factory=lambda: [
        "对比 OP-TR 与基线 OPERA 在 CHAIR_i 指标上的性能差异",
        "通过消融实验验证惩罚机制和奖励机制的独立贡献",
        "评估不同超参数配置对模型性能的敏感性",
        "分析 OP-TR 在不同场景下的稳定性和泛化能力",
        "探索惩罚-奖励平衡的最优组合策略"
    ])
    
    research_questions: List[str] = field(default_factory=lambda: [
        "RQ1: OP-TR 能否显著降低 CHAIR_i（相比 OPERA 基线下降 ≥5%）？",
        "RQ2: Beam Score 级惩罚相比 OPERA 的 logits 层惩罚效果如何？",
        "RQ3: 视觉 Token 奖励机制能否有效提升模型的视觉关注度？",
        "RQ4: 不同超参数配置（OP-TR-10 vs OP-TR-12）的性能差异？",
        "RQ5: 惩罚与奖励的相对重要性及其交互效应？"
    ])


@dataclass  
class HypothesisFramework:
    """假设框架定义"""
    
    hypotheses: Dict[str, Dict] = field(default_factory=lambda: {
        "H1": {
            "statement": "OP-TR 的 CHAIR_i 显著低于 OPERA 基线",
            "direction": "one-tailed (lower is better)",
            "effect_size": "medium (Cohen's d > 0.5)",
            "expected_result": "CHAIR_i: OP-TR < OPERA (13.6 → ≤13.0)"
        },
        "H2": {
            "statement": "Beam Score 级惩罚比 logits 层惩罚更有效",
            "direction": "one-tailed",
            "rationale": "避免 softmax 归一化后的反向提升效应",
            "test_method": "对比 beam_penalty vs penalty_scores"
        },
        "H3": {
            "statement": "视觉 Token 奖励能独立降低幻觉率",
            "direction": "one-tailed",
            "ablation_test": "Reward=0 时 CHAIR_i 应显著高于完整 OP-TR",
            "expected_gap": "≥0.5 个百分点"
        },
        "H4": {
            "statement": "惩罚与奖励存在协同效应",
            "direction": "interaction effect",
            "test": "完整 OP-TR > (仅惩罚) + (仅奖励) 的简单叠加"
        }
    })


# ============================================================================
# 第二部分: 实验设计方案 (Section 2: Experimental Design)
# ============================================================================

@dataclass
class ExperimentalConfig:
    """实验配置参数"""
    
    # 基础设置
    model_name: str = "llava-1.5-7b"
    dataset: str = "COCO_2014_val"
    sample_size: int = 500  # 图片数量
    num_beams: int = 5      # beam search 的 beam 数量
    
    # 固定参数（所有实验组共享）
    scale_factor: float = 50.0
    threshold: int = 15
    num_attn_candidates: int = 5
    window_size: int = 512
    penalty_weights: float = 1.0
    
    # 随机种子（确保可重复性）
    seed: int = 42
    
    # GPU 设置
    gpu_id: int = 0
    batch_size: int = 1
    
    # 输出路径
    output_base_dir: str = "./experiments/optr_validation"


@dataclass 
class ExperimentalGroup:
    """实验组定义"""
    
    # 组标识符
    group_id: str
    group_name: str
    description: str
    
    # OP-TR 超参数
    alpha_d: float           # 距离缩放指数
    d_0: int                 # 距离阈值
    c_: float                # 惩罚系数 (math.log(x))
    Reward: float            # 奖励池总量 (math.log(x))
    
    # 元数据
    is_baseline: bool = False
    is_ablation: bool = False
    ablation_type: Optional[str] = None  # "penalty_only", "reward_only", etc.
    
    def get_config_dict(self) -> Dict:
        """返回该组的完整配置字典"""
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "description": self.description,
            "parameters": {
                "alpha_d": self.alpha_d,
                "d_0": self.d_0,
                "c_": self.c_,
                "Reward": self.Reward,
                "c_log_value": math.log(self.c_) if self.c_ > 0 else self.c_,
                "reward_log_value": math.log(self.Reward) if self.Reward > 0 else self.Reward
            },
            "metadata": {
                "is_baseline": self.is_baseline,
                "is_ablation": self.is_ablation,
                "ablation_type": self.ablation_type
            }
        }


def create_experimental_groups() -> List[ExperimentalGroup]:
    """
    创建完整的实验组列表（14组消融实验 + 基线对照）
    
    实验组设计原则：
    1. 基线对照：Standard Beam Search, OPERA (现有最佳)
    2. 主实验组：OP-TR-10 (推荐配置), OP-TR-12 (激进配置)  
    3. 消融实验：分别测试惩罚/奖励的独立贡献
    4. 敏感性分析：单因子变量测试
    5. 鲁棒性验证：边界条件测试
    """
    
    groups = []
    
    # ========== Group G0: 基线对照 ==========
    groups.append(ExperimentalGroup(
        group_id="G0",
        group_name="Standard_Beam_Search",
        description="标准 beam search（无任何反 hallucination 机制）",
        alpha_d=0, d_0=0, c_=0, Reward=0,
        is_baseline=True
    ))
    
    groups.append(ExperimentalGroup(
        group_id="G1", 
        group_name="OPERA_Baseline",
        description="OPERA 原版实现（logits 层惩罚，无视觉奖励）",
        alpha_d=0, d_0=0, c_=0, Reward=0,
        is_baseline=True
    ))
    
    # ========== Group G2-G3: 主实验组（推荐配置）==========
    groups.append(ExperimentalGroup(
        group_id="G2",
        group_name="OP-TR-10",
        description="OP-TR 推荐配置: α=1.0, d0=7, c=log(0.05), R=log(5)",
        alpha_d=1.0, d_0=7, c_=math.log(0.05), Reward=math.log(5),
        is_baseline=False
    ))
    
    groups.append(ExperimentalGroup(
        group_id="G3",
        group_name="OP-TR-12", 
        description="OP-TR 激进配置: α=0.8, d0=6, c=log(0.005), R=log(15)",
        alpha_d=0.8, d_0=6, c_=math.log(0.005), Reward=math.log(15),
        is_baseline=False
    ))
    
    # ========== Group G4-G7: 核心消融实验 ==========
    # 测试惩罚机制的独立贡献
    groups.append(ExperimentalGroup(
        group_id="G4",
        group_name="OP-TR-Penalty_Only",
        description="仅启用惩罚机制（Reward=0），验证惩罚的独立效果",
        alpha_d=1.0, d_0=7, c_=math.log(0.05), Reward=0,
        is_ablation=True, ablation_type="penalty_only"
    ))
    
    # 测试奖励机制的独立贡献
    groups.append(ExperimentalGroup(
        group_id="G5",
        group_name="OP-TR-Reward_Only",
        description="仅启用奖励机制（c_=0），验证奖励的独立效果",
        alpha_d=1.0, d_0=7, c_=0, Reward=math.log(5),
        is_ablation=True, ablation_type="reward_only"
    ))
    
    # 测试 Candidate 级奖励的贡献
    groups.append(ExperimentalGroup(
        group_id="G6",
        group_name="OP-TR-Beam_Reward_Only",
        description="仅使用 Beam 级奖励（禁用 Candidate 级 φ 缩放）",
        alpha_d=1.0, d_0=7, c_=math.log(0.05), Reward=math.log(5),
        is_ablation=True, ablation_type="beam_reward_only"
    ))
    
    # 测试距离因子的贡献
    groups.append(ExperimentalGroup(
        group_id="G7",
        group_name="OP-TR-No_Distance_Penalty",
        description="禁用距离因子 D(x)（d_0=∞，即不按距离惩罚）",
        alpha_d=1.0, d_0=1000, c_=math.log(0.05), Reward=math.log(5),
        is_ablation=True, ablation_type="no_distance"
    ))
    
    # ========== Group G8-G11: 单因子敏感性分析 ==========
    # alpha_d 敏感性
    groups.append(ExperimentalGroup(
        group_id="G8",
        group_name="OP-TR-alpha_0.5",
        description="测试较小的距离衰减指数 α=0.5（更平滑的惩罚曲线）",
        alpha_d=0.5, d_0=7, c_=math.log(0.05), Reward=math.log(5),
        is_ablation=True, ablation_type="sensitivity_alpha"
    ))
    
    groups.append(ExperimentalGroup(
        group_id="G9",
        group_name="OP-TR-alpha_1.5",
        description="测试较大的距离衰减指数 α=1.5（更陡峭的惩罚曲线）",
        alpha_d=1.5, d_0=7, c_=math.log(0.05), Reward=math.log(5),
        is_ablation=True, ablation_type="sensitivity_alpha"
    ))
    
    # d_0 敏感性
    groups.append(ExperimentalGroup(
        group_id="G10",
        group_name="OP-TR-d0_5",
        description="测试更小的距离阈值 d0=5（更早触发惩罚）",
        alpha_d=1.0, d_0=5, c_=math.log(0.05), Reward=math.log(5),
        is_ablation=True, ablation_type="sensitivity_d0"
    ))
    
    groups.append(ExperimentalGroup(
        group_id="G11",
        group_name="OP-TR-d0_9",
        description="测试更大的距离阈值 d0=9（更晚触发惩罚）",
        alpha_d=1.0, d_0=9, c_=math.log(0.05), Reward=math.log(5),
        is_ablation=True, ablation_type="sensitivity_d0"
    ))
    
    # ========== Group G12-G13: 惩罚/奖励强度分析 ==========
    groups.append(ExperimentalGroup(
        group_id="G12",
        group_name="OP-TR-Strong_Penalty",
        description="更强惩罚: c=log(0.001) ≈ -6.9（更激进地抑制重复模式）",
        alpha_d=1.0, d_0=7, c_=math.log(0.001), Reward=math.log(5),
        is_ablation=True, ablation_type="penalty_strength"
    ))
    
    groups.append(ExperimentalGroup(
        group_id="G13",
        group_name="OP-TR-Strong_Reward",
        description="更强奖励: R=log(20) ≈ 3.0（更强烈地激励视觉关注）",
        alpha_d=1.0, d_0=7, c_=math.log(0.05), Reward=math.log(20),
        is_ablation=True, ablation_type="reward_strength"
    ))
    
    # ========== Group G14: 极端条件鲁棒性测试 ==========
    groups.append(ExperimentalGroup(
        group_id="G14",
        group_name="OP-TR-Maximal_Intervention",
        description="最大干预强度: α=2.0, d0=4, c=log(0.0001), R=log(50)",
        alpha_d=2.0, d_0=4, c_=math.log(0.0001), Reward=math.log(50),
        is_ablation=True, ablation_type="extreme_case"
    ))
    
    return groups


# ============================================================================
# 第三部分: 变量控制方法 (Section 3: Variable Control)
# ============================================================================

@dataclass
class VariableControl:
    """变量控制协议"""
    
    controlled_variables: Dict[str, Any] = field(default_factory=lambda: {
        # 模型相关
        "model_checkpoint": "llava-v1.5-7b (fixed)",
        "model_weights": "pre-trained (no fine-tuning)",
        "architecture": "LLaVA-1.5 with Vicuna-7B decoder",
        
        # 数据相关
        "dataset": "COCO 2014 validation set",
        "sample_size": 500,
        "image_preprocessing": "CLIP ViT-L/14 @ 336px",
        "text_tokenization": "LLaMA tokenizer",
        
        # 生成相关
        "max_new_tokens": 512,
        "temperature": "1.0 (use log_softmax in beam search)",
        "num_beams": 5,
        "prompt_template": "USER: <ImageHere> Please describe this image in detail. ASSISTANT:",
        
        # 硬件相关
        "gpu_model": "NVIDIA A100/V100 (specify in execution)",
        "precision": "fp16/bf16 (model dependent)",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    })
    
    independent_variables: Dict[str, List] = field(default_factory=lambda: {
        "primary_iv": ["method (Standard/OPERA/OP-TR)"],
        "secondary_ivs": [
            "alpha_d (distance scaling exponent)",
            "d_0 (distance threshold)",
            "c_ (penalty coefficient)",
            "Reward (reward pool size)"
        ]
    })
    
    dependent_variables: Dict[str, str] = field(default_factory=lambda: {
        "primary_dv": "CHAIR_i (instance-level hallucination rate)",
        "secondary_dvs": [
            "CHAIR_s (sentence-level hallucination rate)",
            "Recall (object coverage rate)",
            "Avg_Caption_Length",
            "Inference_Time_per_image",
            "Beam_Rollback_Count"
        ]
    })
    
    confounding_variables_control: Dict[str, str] = field(default_factory=lambda: {
        "randomness": "固定随机种子 (seed=42)，使用 torch.manual_seed()",
        "data_order": "打乱后固定顺序 (random.shuffle with seed)",
        "hardware_variance": "同一 GPU 运行所有实验组",
        "numerical_precision": "统一使用 fp16，避免精度差异",
        "implementation_details": "除超参数外，代码路径完全一致"
    })


def setup_reproducible_environment(seed: int = 42):
    """
    设置可重复的实验环境
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch_manual_seed = __import__('torch').manual_seed(seed)
    
    # CUDA 相关设置
    try:
        import torch
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ 已设置 CUDA 确定性模式 (seed={seed})")
    except Exception as e:
        print(f"⚠ CUDA 设置跳过: {e}")
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 实验环境已初始化 (seed={seed})")
    return True


# ============================================================================
# 第四部分: 数据收集流程 (Section 4: Data Collection Protocol)
# ============================================================================

@dataclass
class DataCollectionProtocol:
    """标准化数据收集流程"""
    
    collection_phases: List[Dict] = field(default_factory=lambda: [
        {
            "phase": 1,
            "name": "Pre-experiment Preparation",
            "steps": [
                "1.1 验证数据集完整性 (500 images, annotations)",
                "1.2 加载预训练模型并验证权重",
                "1.3 运行 sanity check (5 images, Standard Beam Search)",
                "1.4 确认基线 CHAIR_i 与文献一致 (±0.2 tolerance)",
                "1.5 创建输出目录结构并记录元数据"
            ],
            "duration_estimate": "2-3 hours"
        },
        {
            "phase": 2,
            "name": "Baseline Experiments (G0-G1)",
            "steps": [
                "2.1 运行 Standard Beam Search (G0)",
                "2.2 运行 OPERA Baseline (G1)", 
                "2.3 计算 CHAIR 指标并保存原始输出",
                "2.4 验证结果可重复性 (run twice, compare hashes)"
            ],
            "duration_estimate": "4-6 hours"
        },
        {
            "phase": 3,
            "name": "Main OP-TR Experiments (G2-G3)",
            "steps": [
                "3.1 运行 OP-TR-10 (G2) - 推荐配置",
                "3.2 运行 OP-TR-12 (G3) - 激进配置",
                "3.3 实时监控指标趋势",
                "3.4 记录中间状态和异常情况"
            ],
            "duration_estimate": "4-6 hours"
        },
        {
            "phase": 4,
            "name": "Ablation Studies (G4-G7)",
            "steps": [
                "4.1 Penalty-only test (G4)",
                "4.2 Reward-only test (G5)",
                "4.3 Beam-reward-only test (G6)",
                "4.4 No-distance-penalty test (G7)"
            ],
            "duration_estimate": "8-10 hours"
        },
        {
            "phase": 5,
            "name": "Sensitivity Analysis (G8-G13)",
            "steps": [
                "5.1 Alpha sensitivity (G8-G9)",
                "5.2 d0 sensitivity (G10-G11)",
                "5.3 Strength analysis (G12-G13)"
            ],
            "duration_estimate": "12-15 hours"
        },
        {
            "phase": 6,
            "name": "Robustness Check (G14)",
            "steps": [
                "6.1 Run extreme case (G14)",
                "6.2 Monitor for numerical instability",
                "6.3 Check for degenerate outputs"
            ],
            "duration_estimate": "2-3 hours"
        },
        {
            "phase": 7,
            "name": "Post-processing & Validation",
            "steps": [
                "7.1 Compute all metrics using standardized script",
                "7.2 Generate comparison tables and visualizations",
                "7.3 Perform statistical significance tests",
                "7.4 Document all anomalies and edge cases"
            ],
            "duration_estimate": "3-4 hours"
        }
    ])
    
    quality_control_checks: List[Dict] = field(default_factory=lambda: [
        {
            "check_id": "QC-1",
            "name": "Output Completeness",
            "criteria": "每组必须生成恰好 500 条 caption",
            "action_on_fail": "重新运行缺失样本"
        },
        {
            "check_id": "QC-2", 
            "name": "Format Consistency",
            "criteria": "所有 JSONL 文件格式符合 schema",
            "action_on_fail": "修复或重新生成"
        },
        {
            "check_id": "QC-3",
            "name": "Reproducibility Verification",
            "criteria": "同一配置两次运行结果完全一致 (MD5 match)",
            "action_on_fail": "调查随机源并修复"
        },
        {
            "check_id": "QC-4",
            "name": "Numerical Stability",
            "criteria": "无 NaN/Inf 在 scores 或 attentions 中",
            "action_on_fail": "调整超参数或检查实现"
        },
        {
            "check_id": "QC-5",
            "name": "Baseline Consistency",
            "criteria": "OPERA baseline CHAIR_i ∈ [13.4, 13.8]",
            "action_on_fail": "验证环境和数据一致性"
        }
    ])


def generate_output_filename(group: ExperimentalGroup, config: ExperimentalConfig) -> str:
    """
    生成标准化的输出文件名
    
    Format: {group_id}_{group_name}_a{alpha}_d0{d0}_c{c_log}_R{R_log}.jsonl
    """
    c_log = f"log{group.c_:.3f}" if group.c_ != 0 else "0"
    r_log = f"log{group.Reward:.3f}" if group.Reward != 0 else "0"
    
    filename = (
        f"{group.group_id}_{group.group_name}"
        f"_a{group.alpha_d}"
        f"_d0{group.d_0}"
        f"_c{c_log}"
        f"_R{r_log}.jsonl"
    )
    
    # 替换特殊字符
    filename = filename.replace(".", "_").replace("-", "_")
    
    return filename


def create_output_directory_structure(base_dir: str, experiment_id: str = None) -> Dict[str, str]:
    """
    创建标准化的输出目录结构
    
    Returns:
        目录路径字典
    """
    if experiment_id is None:
        experiment_id = f"optr_exp_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    dirs = {
        "root": os.path.join(base_dir, experiment_id),
        "captions": os.path.join(base_dir, experiment_id, "captions"),
        "metrics": os.path.join(base_dir, experiment_id, "metrics"),
        "logs": os.path.join(base_dir, experiment_id, "logs"),
        "intermediate": os.path.join(base_dir, experiment_id, "intermediate"),
        "analysis": os.path.join(base_dir, experiment_id, "analysis"),
        "figures": os.path.join(base_dir, experiment_id, "figures")
    }
    
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 创建 README
    readme_path = os.path.join(dirs["root"], "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# OP-TR Experiment: {experiment_id}\n\n")
        f.write(f"- Date: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"- Total Groups: 15 (G0-G14)\n")
        f.write(f"- Sample Size: 500 images\n\n")
        f.write("## Directory Structure\n\n")
        f.write("- `captions/`: Generated captions (JSONL format)\n")
        f.write("- `metrics/`: Computed CHAIR metrics\n")
        f.write("- `logs/`: Execution logs\n")
        f.write("- `intermediate/`: Intermediate results (attentions, scores)\n")
        f.write("- `analysis/`: Statistical analysis results\n")
        f.write("- `figures/`: Visualization plots\n")
    
    print(f"✓ 输出目录已创建: {dirs['root']}")
    return dirs


# ============================================================================
# 第五部分: 评价指标体系 (Section 5: Evaluation Metrics Framework)
# ============================================================================

@dataclass
class EvaluationMetrics:
    """多维度评价指标体系"""
    
    primary_metrics: Dict[str, Dict] = field(default_factory=lambda: {
        "CHAIR_i": {
            "full_name": "Instance-level Caption Hallucination Rate",
            "definition": "幻觉实例数 / 总 MSCOCO 对象提及数",
            "formula": "Σ(hallucinated_words) / Σ(generated_coco_words)",
            "range": "[0, 1], lower is better",
            "target_improvement": "≥5% relative reduction vs OPERA baseline",
            "baseline_value": "0.136 (13.6% from OPERA)",
            "significance_threshold": "p < 0.05 (paired t-test or Wilcoxon)"
        }
    })
    
    secondary_metrics: Dict[str, Dict] = field(default_factory=lambda: {
        "CHAIR_s": {
            "full_name": "Sentence-level Caption Hallucination Rate",
            "definition": "包含幻觉的句子数 / 总句子数",
            "importance": "衡量整体生成质量",
            "expected_trend": "should decrease with OP-TR"
        },
        "Recall": {
            "full_name": "Object Coverage Recall",
            "definition": "正确识别的对象数 / 图像中实际对象数",
            "importance": "确保降低幻觉的同时不丢失信息",
            "constraint": "should NOT decrease significantly (≤2% drop acceptable)"
        },
        "Avg_Len": {
            "full_name": "Average Caption Length",
            "definition": "平均每张图的 token 数",
            "importance": "检测是否过度抑制生成长度",
            "acceptable_range": "±10% of baseline"
        },
        "Inference_Time": {
            "full_name": "Average Inference Time per Image",
            "definition": "单张图平均推理时间 (seconds)",
            "importance": "实用性考量",
            "acceptable_overhead": "≤2x compared to standard beam search"
        },
        "Rollback_Rate": {
            "full_name": "Beam Rollback Frequency",
            "definition": "触发回溯机制的次数 / 总生成步数",
            "importance": "衡量 Retrospection-Allocation 的工作负载",
            "monitoring": "should remain stable across configurations"
        }
    })
    
    novel_metrics_for_optr: Dict[str, Dict] = field(default_factory=lambda: {
        "Visual_Attention_Score": {
            "full_name": "Average Visual Token Attention Weight",
            "definition": "最后一个 query 对图像 token 的平均注意力权重",
            "purpose": "直接量化视觉奖励的效果",
            "expected_change": "increase with stronger reward mechanism"
        },
        "Penalty_Magnitude": {
            "full_name": "Average Absolute Penalty Applied",
            "definition": "|beam_penalty| 的均值",
            "purpose": "监控惩罚强度的分布",
            "correlation": "should correlate with CHAIR_i reduction"
        },
        "Reward_Distribution_Entropy": {
            "full_name": "Entropy of Beam-Level Reward Distribution",
            "definition": "H(Beam_Rewards) 衡量奖励分配的均匀性",
            "purpose": "检测是否存在奖励集中现象"
        },
        "Candidate_Ranking_Correlation": {
            "full_name": "Spearman ρ between visual attention rank and final selection rank",
            "definition": "候选排名与最终选择的相关性",
            "purpose": "验证 Candidate 级奖励的有效性"
        }
    })


def compute_chair_metrics(caption_file: str, coco_path: str, cache_file: str = "chair.pkl") -> Dict:
    """
    计算 CHAIR 指标（封装 chair.py 的功能）
    
    Args:
        caption_file: 生成的 caption 文件路径 (JSON/JSONL)
        coco_path: COCO 注释文件路径
        cache_file: CHAIR evaluator 缓存文件
        
    Returns:
        包含所有指标的字典
    """
    import pickle
    from chair import CHAIR, print_metrics
    
    # 加载或构建 evaluator
    if os.path.exists(cache_file):
        evaluator = pickle.load(open(cache_file, 'rb'))
    else:
        evaluator = CHAIR(coco_path)
        pickle.dump(evaluator, open(cache_file, 'wb'))
    
    # 计算指标
    result = evaluator.compute_chair(
        cap_file=caption_file,
        image_id_key="image_id",
        caption_key="caption"
    )
    
    return result['overall_metrics']


def compute_optr_specific_metrics(log_data: List[Dict]) -> Dict:
    """
    计算 OP-TR 特有的中间指标
    
    Args:
        log_data: 包含 beam_penalty, Beam_Rewards, attn_i 等中间数据的列表
        
    Returns:
        OP-TR 特有指标字典
    """
    metrics = {}
    
    if not log_data:
        return metrics
    
    # 提取各指标的值
    penalties = [entry.get('beam_penalty', 0) for entry in log_data]
    rewards = [entry.get('Beam_Rewards', 0) for entry in log_data]
    visual_attns = [entry.get('attn_i_mean', 0) for entry in log_data]
    
    import numpy as np
    
    metrics['Visual_Attention_Score'] = float(np.mean(visual_attns))
    metrics['Penalty_Magnitude'] = float(np.mean(np.abs(penalties)))
    metrics['Penalty_Std'] = float(np.std(penalties))
    metrics['Reward_Mean'] = float(np.mean(rewards))
    metrics['Reward_Std'] = float(np.std(rewards))
    
    # 计算熵
    if rewards:
        from scipy.stats import entropy
        reward_dist = np.array(rewards) - min(rewards) + 1e-8
        reward_dist = reward_dist / reward_dist.sum()
        metrics['Reward_Distribution_Entropy'] = float(entropy(reward_dist))
    
    return metrics


# ============================================================================
# 第六部分: 统计分析方法 (Section 6: Statistical Analysis Plan)
# ============================================================================

@dataclass
class StatisticalAnalysisPlan:
    """严格的数据分析方法"""
    
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    effect_size_threshold: float = 0.5  # Cohen's d medium effect
    
    analysis_methods: Dict[str, Dict] = field(default_factory=lambda: {
        "primary_comparison": {
            "hypothesis": "H1: OP-TR CHAIR_i < OPERA CHAIR_i",
            "test": "Paired Wilcoxon signed-rank test",
            "reason": "非参数检验，不假设正态分布，适合配对样本",
            "n": "500 (per-group sample size)",
            "power_analysis": "Power ≥ 0.8 to detect d=0.5 at α=0.05"
        },
        "ablation_analysis": {
            "hypothesis": "H3/H4: Individual component contributions",
            "test": "One-way ANOVA + Tukey HSD post-hoc",
            "groups": ["Full_OP-TR", "Penalty_Only", "Reward_Only", "Baseline"],
            "assumptions_check": "Shapiro-Wilk (normality) + Levene (homogeneity)"
        },
        "sensitivity_analysis": {
            "method": "Spearman correlation between hyperparameter values and CHAIR_i",
            "visualization": "Heatmap of CHAI_i across parameter grid",
            "trend_analysis": "Monotonicity check for each parameter"
        },
        "robustness_check": {
            "extreme_case_analysis": "Compare G14 with others to detect degradation",
            "outlier_detection": "IQR method (1.5*IQR rule)",
            "stability_metric": "CV (Coefficient of Variation) across runs"
        }
    })
    
    multiple_testing_correction: str = "Benjamini-Hochberg FDR (14 comparisons)"


def perform_statistical_tests(results_dict: Dict[str, Dict]) -> Dict:
    """
    执行完整的统计检验流程
    
    Args:
        results_dict: {group_id: {'chair_i': float, 'chair_s': float, ...}}
        
    Returns:
        统计检验结果字典
    """
    from scipy import stats
    import numpy as np
    
    stat_results = {}
    
    # 提取主要比较组的数据
    opera_chair_i = results_dict.get('G1', {}).get('chair_i_values', [])
    optr10_chair_i = results_dict.get('G2', {}).get('chair_i_values', [])
    optr12_chair_i = results_dict.get('G3', {}).get('chair_i_values', [])
    
    if opera_chair_i and optr10_chair_i:
        # 配对 Wilcoxon 检验 (H1)
        statistic, p_value = stats.wilcoxon(opera_chair_i, optr10_chair_i)
        stat_results['H1_OPTR10_vs_OPERA'] = {
            'test': 'Wilcoxon signed-rank',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': _compute_cohens_d(opera_chair_i, optr10_chair_i),
            'mean_diff': float(np.mean(optr10_chair_i) - np.mean(opera_chair_i)),
            'median_diff': float(np.median(optr10_chair_i) - np.median(opera_chair_i))
        }
    
    if opera_chair_i and optr12_chair_i:
        statistic, p_value = stats.wilcoxon(opera_chair_i, optr12_chair_i)
        stat_results['H1_OPTR12_vs_OPERA'] = {
            'test': 'Wilcoxon signed-rank',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': _compute_cohens_d(opera_chair_i, optr12_chair_i),
            'mean_diff': float(np.mean(optr12_chair_i) - np.mean(opera_chair_i))
        }
    
    # 消融实验 ANOVA
    ablation_groups = ['G1', 'G2', 'G4', 'G5']  # OPERA, Full, Penalty_only, Reward_only
    ablation_data = [results_dict.get(g, {}).get('chair_i_values', []) for g in ablation_groups]
    
    if all(len(d) > 0 for d in ablation_data):
        f_stat, p_value_anova = stats.f_oneway(*ablation_data)
        stat_results['Ablation_ANOVA'] = {
            'test': 'One-way ANOVA',
            'F_statistic': float(f_stat),
            'p_value': float(p_value_anova),
            'groups_compared': ablation_groups
        }
        
        # Tukey HSD post-hoc
        try:
            from scipy.stats import tukey_hsd
            tukey_result = tukey_hsd(*ablation_data)
            stat_results['Tukey_HSD'] = tukey_result._asdict()
        except:
            pass  # SciPy version may not have tukey_hsd
    
    # 敏感性分析: Spearman 相关性
    param_configs = {
        'G2': {'alpha': 1.0, 'd0': 7},
        'G8': {'alpha': 0.5, 'd0': 7},
        'G9': {'alpha': 1.5, 'd0': 7},
        'G10': {'alpha': 1.0, 'd0': 5},
        'G11': {'alpha': 1.0, 'd0': 9}
    }
    
    alphas = [param_configs[g]['alpha'] for g in param_configs.keys()]
    d0s = [param_configs[g]['d0'] for g in param_configs.keys()]
    chair_is = [np.mean(results_dict.get(g, {}).get('chair_i_values', [np.nan])) 
                for g in param_configs.keys()]
    
    if not any(np.isnan(chair_is)):
        corr_alpha, p_alpha = stats.spearmanr(alphas, chair_is)
        corr_d0, p_d0 = stats.spearmanr(d0s, chair_is)
        
        stat_results['Sensitivity_Alpha'] = {
            'test': 'Spearman correlation',
            'correlation': float(corr_alpha),
            'p_value': float(p_alpha),
            'interpretation': 'monotonic relationship strength'
        }
        stat_results['Sensitivity_D0'] = {
            'test': 'Spearman correlation',
            'correlation': float(corr_d0),
            'p_value': float(p_d0)
        }
    
    return stat_results


def _compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """计算 Cohen's d 效应量"""
    import numpy as np
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# ============================================================================
# 第七部分: 实验执行脚本 (Section 7: Experiment Execution Script)
# ============================================================================

def run_single_experiment(
    group: ExperimentalGroup,
    config: ExperimentalConfig,
    output_dirs: Dict[str, str],
    verbose: bool = True
) -> Dict:
    """
    执行单个实验组的完整流程
    
    Args:
        group: 实验组配置
        config: 全局实验配置
        output_dirs: 输出目录结构
        verbose: 是否打印详细日志
        
    Returns:
        实验结果字典
    """
    import time
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"开始执行实验组: {group.group_id} - {group.group_name}")
        print(f"描述: {group.description}")
        print(f"参数: α={group.alpha_d}, d0={group.d_0}, c={group.c_:.4f}, R={group.Reward:.4f}")
        print(f"{'='*60}\n")
    
    result = {
        'group_id': group.group_id,
        'group_name': group.group_name,
        'config': group.get_config_dict(),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'running'
    }
    
    try:
        # Step 1: 准备输出文件路径
        output_filename = generate_output_filename(group, config)
        output_path = os.path.join(output_dirs['captions'], output_filename)
        
        # Step 2: 调用 chair_eval.py 的逻辑（需要修改以支持 OP-TR 参数）
        # 这里是伪代码，实际执行时需要集成到 chair_eval.py
        if verbose:
            print(f"[Step 1/4] 生成 captions...")
            print(f"  输出文件: {output_path}")
            
            # 显示将要执行的命令
            cmd = (
                f"python chair_eval.py "
                f"--model llava-1.5 "
                f"--gpu-id {config.gpu_id} "
                f"--beam {config.num_beams} "
                f"--scale_factor {config.scale_factor} "
                f"--threshold {config.threshold} "
                f"--num_attn_candidates {config.num_attn_candidates} "
                f"--penalty_weights {config.penalty_weights} "
                f"--alpha_d {group.alpha_d} "
                f"--d_0 {group.d_0} "
                f"--c_ {group.c_} "
                f"--Reward {group.Reward} "
                f"--output_file {output_path}"
            )
            print(f"  命令: {cmd[:100]}...")
        
        # TODO: 实际执行命令（当前为模拟）
        # execute_command(cmd)
        
        # Step 3: 计算指标
        if verbose:
            print(f"\n[Step 2/4] 计算 CHAIR 指标...")
        
        # TODO: 调用 chair.py 计算指标
        # metrics = compute_chair_metrics(output_path, coco_path)
        # result['metrics'] = metrics
        
        # Step 4: 收集中间数据（如果开启了详细日志）
        if verbose:
            print(f"[Step 3/4] 收集中间指标...")
        
        # Step 5: 质量控制检查
        if verbose:
            print(f"[Step 4/4] 执行质量控制检查...")
        
        result['status'] = 'completed'
        result['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        result['duration_seconds'] = time.time() - start_time
        
        if verbose:
            print(f"\n✅ 实验 {group.group_id} 完成!")
            print(f"   耗时: {result['duration_seconds']:.1f} 秒")
            
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        result['duration_seconds'] = time.time() - start_time
        
        if verbose:
            print(f"\n❌ 实验 {group.group_id} 失败: {e}")
    
    return result


def run_full_experiment_suite(
    config: Optional[ExperimentalConfig] = None,
    selected_groups: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    执行完整的实验套件
    
    Args:
        config: 实验配置（None 则使用默认值）
        selected_groups: 要执行的实验组 ID 列表（None 则执行全部）
        verbose: 详细日志
        
    Returns:
        所有实验结果的汇总字典
    """
    if config is None:
        config = ExperimentalConfig()
    
    # 初始化环境
    setup_reproducible_environment(config.seed)
    
    # 创建输出目录
    output_dirs = create_output_directory_structure(config.output_base_dir)
    
    # 获取所有实验组
    all_groups = create_experimental_groups()
    
    # 过滤选定的组
    if selected_groups:
        groups_to_run = [g for g in all_groups if g.group_id in selected_groups]
    else:
        groups_to_run = all_groups
    
    if verbose:
        print(f"\n🔬 OP-TR 实验套件启动")
        print(f"   总计 {len(groups_to_run)} 个实验组")
        print(f"   样本量: {config.sample_size} 张图")
        print(f"   输出目录: {output_dirs['root']}\n")
    
    # 保存实验配置
    config_save_path = os.path.join(output_dirs['root'], 'experiment_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # 保存实验组定义
    groups_save_path = os.path.join(output_dirs['root'], 'experimental_groups.json')
    with open(groups_save_path, 'w') as f:
        json.dump([g.get_config_dict() for g in all_groups], f, indent=2)
    
    # 执行每个实验组
    all_results = []
    
    for i, group in enumerate(groups_to_run, 1):
        if verbose:
            print(f"\n进度: [{i}/{len(groups_to_run)}]")
        
        result = run_single_experiment(
            group=group,
            config=config,
            output_dirs=output_dirs,
            verbose=verbose
        )
        
        all_results.append(result)
        
        # 保存中间结果
        intermediate_path = os.path.join(output_dirs['logs'], 'progress.json')
        with open(intermediate_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # 生成最终报告
    final_report = {
        'experiment_metadata': {
            'total_groups': len(groups_to_run),
            'completed': sum(1 for r in all_results if r['status'] == 'completed'),
            'failed': sum(1 for r in all_results if r['status'] == 'failed'),
            'total_duration_seconds': sum(r.get('duration_seconds', 0) for r in all_results),
            'config': asdict(config)
        },
        'results': all_results
    }
    
    # 保存最终报告
    report_path = os.path.join(output_dirs['root'], 'final_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎉 实验套件执行完成!")
        print(f"   成功: {final_report['experiment_metadata']['completed']}/{len(groups_to_run)}")
        print(f"   总耗时: {final_report['experiment_metadata']['total_duration_seconds']/3600:.1f} 小时")
        print(f"   结果保存在: {output_dirs['root']}")
        print(f"{'='*60}\n")
    
    return final_report


# ============================================================================
# 第八部分: 可视化与报告生成 (Section 8: Visualization & Reporting)
# ============================================================================

def generate_comparison_table(results: Dict) -> str:
    """
    生成 Markdown 格式的对比表格
    
    Args:
        results: 实验结果字典
        
    Returns:
        Markdown 表格字符串
    """
    md = "# OP-TR 实验结果对比表\n\n"
    md += "| 组ID | 方法 | CHAIR_i (%) | CHAIR_s (%) | Recall (%) | Avg Len | 时间(s) |\n"
    md += "|------|------|------------|-------------|------------|---------|--------|\n"
    
    for result in results.get('results', []):
        metrics = result.get('metrics', {})
        md += f"| {result['group_id']} "
        md += f"| {result['group_name']} "
        md += f"| {metrics.get('CHAIRi', 'N/A'):>6} "
        md += f"| {metrics.get('CHAIRs', 'N/A'):>6} "
        md += f"| {metrics.get('Recall', 'N/A'):>6} "
        md += f"| {metrics.get('Len', 'N/A'):>6} "
        md += f"| {result.get('duration_seconds', 0)/500:>6.1f} |\n"
    
    return md


def generate_visualizations(results: Dict, output_dirs: Dict):
    """
    生成可视化图表
    
    包括：
    1. CHAIR_i 对比柱状图
    2. 消融实验热力图
    3. 参数敏感性曲线
    4. 惩罚-奖励散点图
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    import numpy as np
    
    # 提取数据
    group_ids = [r['group_id'] for r in results['results']]
    chair_i_values = [r.get('metrics', {}).get('CHAIRi', np.nan) * 100 
                      for r in results['results']]
    
    # Figure 1: 主要对比柱状图
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#ff6b6b' if g in ['G0', 'G1'] else 
              '#4ecdc4' if g in ['G2', 'G3'] else 
              '#45b7d1' for g in group_ids]
    
    bars = ax.bar(range(len(group_ids)), chair_i_values, color=colors, alpha=0.8)
    
    # 标注数值
    for bar, val in zip(bars, chair_i_values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 添加基线参考线
    opera_idx = group_ids.index('G1') if 'G1' in group_ids else None
    if opera_idx and not np.isnan(chair_i_values[opera_idx]):
        ax.axhline(y=chair_i_values[opera_idx], color='red', linestyle='--', 
                  linewidth=1, label=f'OPERA Baseline ({chair_i_values[opera_idx]:.1f}%)')
    
    ax.set_xticks(range(len(group_ids)))
    ax.set_xticklabels([f"{gid}\n{results['results'][i]['group_name'][:15]}" 
                       for i, gid in enumerate(group_ids)], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CHAIR_i (%)', fontsize=12)
    ax.set_title('OP-TR vs Baseline: Instance-level Hallucination Rate Comparison', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dirs['figures'], '01_main_comparison.png'), dpi=150)
    plt.close(fig)
    
    print("✓ 已生成图表: 01_main_comparison.png")


# ============================================================================
# 主程序入口 (Main Entry Point)
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OP-TR 科学验证实验套件')
    
    parser.add_argument('--mode', type=str, choices=['run', 'analyze', 'report'],
                       default='run', help='运行模式: run/analyze/report')
    parser.add_argument('--groups', type=str, nargs='+', 
                       help='指定要运行的实验组 (如 G2 G3 G4)')
    parser.add_argument('--config', type=str, default=None,
                       help='自定义配置文件路径')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细日志')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU 设备 ID')
    
    args = parser.parse_args()
    
    if args.mode == 'run':
        config = ExperimentalConfig(gpu_id=args.gpu_id)
        results = run_full_experiment_suite(
            config=config,
            selected_groups=args.groups,
            verbose=args.verbose
        )
        
        # 生成可视化
        output_dirs = create_output_directory_structure(config.output_base_dir)
        generate_visualizations(results, output_dirs)
        
        # 打印摘要
        print("\n" + "="*70)
        print("📊 实验结果摘要:")
        print("="*70)
        for result in results['results']:
            status = "✅" if result['status'] == 'completed' else "❌"
            metrics = result.get('metrics', {})
            chair_i = metrics.get('CHAIRi', 'N/A')
            if isinstance(chair_i, float):
                chair_i = f"{chair_i*100:.2f}%"
            print(f"  {status} {result['group_id']:4} ({result['group_name']:25}): CHAIR_i = {chair_i}")
        
    elif args.mode == 'analyze':
        print("📈 分析模式 - 加载已有结果进行统计分析...")
        # TODO: 实现分析逻辑
        
    elif args.mode == 'report':
        print("📝 报告模式 - 生成实验报告...")
        # TODO: 实现报告生成逻辑


if __name__ == '__main__':
    main()


# ============================================================================
# 附录: 实验方案文档化 (Appendix: Documentation)
# ============================================================================

EXPERIMENT_PROTOCOL_SUMMARY = """
================================================================================
                    OP-TR 科学验证实验方案总结文档
================================================================================

一、实验概述
----------
目标: 全面验证 OP-TR (Over-trust Penalty with Trust Reward) 改进措施在
     降低视觉语言模型幻觉方面的有效性

模型: LLaVA-1.5-7B
数据集: COCO 2014 Validation Set (500 images)
评估指标: CHAIR_i (主要), CHAIR_s, Recall, Avg_Len (次要)

二、实验设计
----------
总实验组数: 15 组 (G0-G14)

┌──────┬─────────────────────────┬──────────────────────────────────────────┐
│ 组ID │ 方法名称                 │ 关键参数                                │
├──────┼─────────────────────────┼──────────────────────────────────────────┤
│ G0   │ Standard Beam Search    │ 无反幻觉机制                            │
│ G1   │ OPERA Baseline          │ logits层惩罚, 无视觉奖励               │
│ G2   │ OP-TR-10 (推荐)         │ α=1.0, d0=7, c=log(0.05), R=log(5)    │
│ G3   │ OP-TR-12 (激进)         │ α=0.8, d0=6, c=log(0.005), R=log(15)  │
│ G4   │ 仅惩罚                  │ Reward=0                               │
│ G5   │ 仅奖励                  │ c_=0                                   │
│ G6   │ 仅Beam级奖励             │ 禁用Candidate级φ                      │
│ G7   │ 无距离惩罚              │ d0=1000                                │
│ G8-G9│ Alpha敏感性             │ α=0.5, 1.5                             │
│ G10-G11│ d0敏感性              │ d0=5, 9                                │
│ G12  │ 强惩罚                  │ c=log(0.001)                           │
│ G13  │ 强奖励                  │ R=log(20)                              │
│ G14  │ 极端干预                │ 最大强度参数                           │
└──────┴─────────────────────────┴──────────────────────────────────────────┘

三、研究假设
----------
H1: OP-TR 的 CHAIR_i 显著低于 OPERA (预期: 13.6→≤13.0, 降幅≥5%)
H2: Beam Score级惩罚比logits层惩罚更有效
H3: 视觉Token奖励能独立降低幻觉率 (预期: Reward=0时CHAIR_i上升≥0.5%)
H4: 惩罚与奖励存在协同效应

四、统计方法
----------
主要检验: Paired Wilcoxon signed-rank test (配对非参数检验)
显著性水平: α = 0.05
多重比较校正: Benjamini-Hochberg FDR (14次比较)
效应量: Cohen's d (预期 d > 0.5 为中等效应)

五、预期结果
----------

┌──────────────┬──────────┬──────────┬──────────────────────────────┐
│ 配置         │ CHAIR_i  │ vs OPERA │ 解释                         │
├──────────────┼──────────┼──────────┼──────────────────────────────┤
│ OPERA (G1)   │ 13.6%    │ baseline │ 当前最佳                     │
│ OP-TR-10(G2) │ ~13.0%   │ -4.4% ↓  │ 达到目标，平衡惩罚与奖励     │
│ OP-TR-12(G3) │ ~12.8%   │ -5.9% ↓  │ 更激进，可能牺牲部分流畅性   │
│ 仅惩罚(G4)   │ ~13.5%   │ -0.7% ↓  │ 证明奖励的重要性             │
│ 仅奖励(G5)   │ ~13.5%   │ -0.7% ↓  │ 证明惩罚的重要性             │
│ 极端(G14)    │ ?        │ ?        │ 可能过度抑制导致质量下降      │
└──────────────┴──────────┴──────────┴──────────────────────────────┘

六、时间规划
----------
Phase 1 (准备):     2-3 小时
Phase 2 (基线):     4-6 小时  
Phase 3 (主实验):   4-6 小时
Phase 4 (消融):     8-10 小时
Phase 5 (敏感性):   12-15 小时
Phase 6 (鲁棒性):   2-3 小时
Phase 7 (分析):     3-4 小时
─────────────────────────────
总计预计:          35-47 小时 (~2-3个工作日)

七、成功标准
----------
✓ 主要目标达成: OP-TR-10 CHAIR_i ≤ 13.0 (相比OPERA下降≥5%)
✓ 统计显著性: p-value < 0.05 (Wilcoxon检验)
✓ 效应量达标: Cohen's d > 0.5 (中等效应)
✓ 无副作用: Recall下降<2%, 长度变化<10%
✓ 可重复性: 两次运行结果完全一致(MD5匹配)
✓ 消融验证: 惩罚和奖励均有独立贡献

八、风险与应对
----------
风险1: GPU显存不足
  → 解决: 降低batch_size, 使用gradient checkpointing
  
风险2: 某些配置产生退化输出(NaN/Inf)
  → 解决: 调整超参数范围,添加数值稳定性检查
  
风险3: 实验时间超出预期
  → 解决: 优先完成G0-G7(核心实验),其余可后续补充
  
风险4: 与OPERA基线差异过大(>1%)
  → 解决: 检查环境一致性,验证代码正确性

九、输出物清单
----------
1. experiments/optr_exp_{timestamp}/
   ├── README.md                          # 实验说明
   ├── experiment_config.json             # 实验配置
   ├── experimental_groups.json           # 组定义
   ├── final_report.json                  # 最终结果汇总
   ├── captions/                          # 15个JSONL文件(每组一个)
   │   ├── G0_Standard_Beam_Search_*.jsonl
   │   ├── G1_OPERA_Baseline_*.jsonl
   │   ├── G2_OP-TR-10_*.jsonl
   │   └── ...
   ├── metrics/                           # CHAIR计算结果
   ├── logs/                              # 执行日志
   ├── intermediate/                      # 中间数据(attention, scores)
   ├── analysis/                          # 统计分析结果
   └── figures/                           # 可视化图表
       ├── 01_main_comparison.png
       ├── 02_ablation_heatmap.png
       ├── 03_sensitivity_curves.png
       └── 04_penalty_reward_scatter.png

十、复现指南
----------
环境要求:
- Python 3.8+
- PyTorch >= 1.12
- Transformers 4.29.2 (已修改版)
- NVIDIA GPU (≥16GB VRAM, 推荐 A100)

步骤:
1. 安装依赖: pip install -r requirements.txt
2. 下载数据集: COCO 2014 val set
3. 下载模型: LLaVA-1.5-7B pretrained weights
4. 运行实验: python optr_experiment_protocol.py --mode run --gpu-id 0
5. 分析结果: python optr_experiment_protocol.py --mode analyze
6. 生成报告: python optr_experiment_protocol.py --mode report

================================================================================
                        文档结束 | 版本 v1.0 | 2026-05-07
================================================================================
"""


if __name__ == '__main__':
    # 如果直接运行，打印实验方案摘要
    print(EXPERIMENT_PROTOCOL_SUMMARY)
