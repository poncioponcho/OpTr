#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OP-TR 实验快速执行脚本
用于运行 G1 (OPERA Baseline) 和 G2 (OP-TR-10) 对比实验
"""

import os
import sys
import json
import math
import time
from datetime import datetime

def run_experiment_group(group_id, group_name, params, data_path, gpu_id=0):
    """
    运行单个实验组
    
    Args:
        group_id: 实验组标识符 (如 'G1', 'G2')
        group_name: 组名称
        params: 参数字典
        data_path: 数据路径
        gpu_id: GPU ID
        
    Returns:
        执行结果字典
    """
    print(f"\n{'='*70}")
    print(f"🔬 开始执行实验组: {group_id} - {group_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # 构建命令
    cmd_parts = [
        "python", "chair_eval.py",
        "--model", "llava-1.5",
        "--gpu-id", str(gpu_id),
        "--beam", "5",
        "--scale_factor", str(params.get('scale_factor', 50)),
        "--threshold", str(params.get('threshold', 15)),
        "--num_attn_candidates", str(params.get('num_attn_candidates', 5)),
        "--penalty_weights", str(params.get('penalty_weights', 1.0)),
        "--data_path", data_path,
    ]
    
    # 如果是 OP-TR，添加额外参数
    if params.get('use_optr'):
        cmd_parts.extend([
            "--use_optr",
            "--alpha_d", str(params['alpha_d']),
            "--d_0", str(params['d_0']),
            "--c_", str(params['c_']),
            "--Reward", str(params['Reward']),
        ])
        
        # 自定义输出文件名
        output_file = f"./log/llava-1.5/{group_id}_{group_name.replace(' ', '_')}.jsonl"
        cmd_parts.extend(["--output_file", output_file])
    
    cmd = " ".join(cmd_parts)
    
    print(f"📋 执行命令:")
    print(f"   {cmd[:100]}...")
    print(f"\n⏱️  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行命令
    import subprocess
    result = {
        'group_id': group_id,
        'group_name': group_name,
        'params': params,
        'start_time': datetime.now().isoformat(),
        'status': 'running',
        'command': cmd,
    }
    
    try:
        # 使用 subprocess 运行
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # 实时输出日志
        print("\n📊 执行日志:")
        print("-" * 70)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"   {output.strip()}")
        
        return_code = process.poll()
        
        end_time = time.time()
        duration = end_time - start_time
        
        result.update({
            'status': 'completed' if return_code == 0 else 'failed',
            'return_code': return_code,
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{duration/3600:.2f}h" if duration > 3600 else f"{duration/60:.1f}min"
        })
        
        print("-" * 70)
        if return_code == 0:
            print(f"✅ 实验 {group_id} 完成! 耗时: {result['duration_formatted']}")
        else:
            stderr_output = process.stderr.read()
            print(f"❌ 实验 {group_id} 失败 (返回码: {return_code})")
            print(f"   错误信息: {stderr_output[:500]}")
            result['error'] = stderr_output
            
    except Exception as e:
        result.update({
            'status': 'error',
            'error': str(e),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time
        })
        print(f"❌ 执行异常: {e}")
    
    return result


def compute_chair_for_group(caption_file, group_id):
    """
    为单个实验组计算 CHAIR 指标
    """
    print(f"\n📈 计算 {group_id} 的 CHAIR 指标...")
    
    try:
        import subprocess
        cmd = f"python chair.py --cap_file {caption_file} --image_id_key image_id --caption_key caption"
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        output, _ = process.communicate()
        
        metrics = {}
        for line in output.split('\n'):
            if ':' in line and any(metric in line for metric in ['CHAIRi', 'CHAIRs', 'Recall', 'Len']):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    metrics[key] = value
        
        print(f"   ✅ 指标计算完成: {metrics}")
        return metrics
        
    except Exception as e:
        print(f"   ❌ 计算失败: {e}")
        return {}


def main():
    """主函数：执行 OP-TR 验证实验"""
    
    print("="*80)
    print("🚀 OP-TR 科学验证实验启动")
    print("="*80)
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 检查环境
    print("\n🔍 环境检查...")
    try:
        import torch
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")
        if torch.cuda.is_available():
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   ⚠ PyTorch 未安装，请先激活 conda 环境: conda activate opera")
        print("   提示: 如果在 macOS 上，可能需要使用 CPU 模式或连接远程 GPU")
        return
    
    # 定义实验配置
    experiments = [
        {
            'group_id': 'G1',
            'group_name': 'OPERA_Baseline',
            'description': 'OPERA 原版实现（logits 层惩罚）',
            'params': {
                'scale_factor': 50.0,
                'threshold': 15,
                'num_attn_candidates': 5,
                'penalty_weights': 1.0,
                'use_optr': False,
            },
            'expected_chair_i': 13.6,
        },
        {
            'group_id': 'G2',
            'group_name': 'OP-TR-10',
            'description': 'OP-TR 推荐配置（Beam级惩罚 + 视觉奖励）',
            'params': {
                'scale_factor': 50.0,
                'threshold': 15,
                'num_attn_candidates': 5,
                'penalty_weights': 1.0,
                'use_optr': True,
                'alpha_d': 1.0,
                'd_0': 7,
                'c_': math.log(0.05),      # ≈ -2.9957
                'Reward': math.log(5),     # ≈ 1.6094
            },
            'expected_chair_i': 13.0,
        }
    ]
    
    # 数据路径
    data_path = "./COCO_2014/val2014/"
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"\n⚠ 数据路径不存在: {data_path}")
        print("   请确认 COCO 2014 val2014 数据集已下载到正确位置")
        print("   或修改 data_path 变量指向正确的路径")
        return
    
    # 创建输出目录
    os.makedirs("./log/llava-1.5", exist_ok=True)
    os.makedirs("./experiments/optr_validation", exist_ok=True)
    
    # 存储结果
    all_results = []
    
    # 执行每个实验组
    for exp in experiments:
        result = run_experiment_group(
            group_id=exp['group_id'],
            group_name=exp['group_name'],
            params=exp['params'],
            data_path=data_path,
            gpu_id=0
        )
        all_results.append(result)
        
        # 如果失败，停止后续实验
        if result['status'] != 'completed':
            print(f"\n⛔ 实验 {exp['group_id']} 失败，终止后续实验")
            break
    
    # 保存执行日志
    log_file = "./experiments/optr_validation/execution_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            'execution_time': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'completed': sum(1 for r in all_results if r['status'] == 'completed'),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("📊 实验执行摘要")
    print("="*80)
    
    for result in all_results:
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        duration = result.get('duration_formatted', 'N/A')
        print(f"{status_icon} {result['group_id']:4} ({result['group_name']:20}): "
              f"状态={result['status']:10}, 耗时={duration}")
    
    print(f"\n📁 详细日志已保存至: {log_file}")
    
    # 如果所有实验都成功，提示下一步
    if all(r['status'] == 'completed' for r in all_results):
        print("\n🎉 所有实验完成！下一步:")
        print("   1. 计算 CHAIR 指标:")
        print("      python chair.py --cap_file ./log/llava-1.5/G1_OPERA_Baseline.jsonl")
        print("      python chair.py --cap_file ./log/llava-1.5/G2_OP-TR-10.jsonl")
        print("   2. 对比结果并生成报告")
        print("   3. 可视化分析")
    
    return all_results


if __name__ == '__main__':
    main()
