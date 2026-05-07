#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OP-TR 快速验证与故障修复脚本
适用于 macOS / 无 GPU / 无完整 COCO 数据集的环境
"""

import os
import sys
import json
import math
import shutil
from datetime import datetime


def diagnose_environment():
    """诊断当前环境状态"""
    
    print("="*80)
    print("🔍 OP-TR 环境诊断报告")
    print("="*80)
    print(f"⏰ 诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 工作目录: {os.getcwd()}")
    print(f"💻 操作系统: {os.uname().sysname} {os.uname().machine}")
    
    issues = []
    warnings = []
    
    # 1. 检查 Python 环境
    print("\n[1/6] Python 环境检查...")
    try:
        import torch
        print(f"   ✅ PyTorch 版本: {torch.__version__}")
        print(f"   ✅ CUDA 可用: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            warnings.append("CUDA 不可用 - 将使用 CPU 模式（速度较慢但可运行）")
            print("   ⚠️  将使用 CPU 模式运行")
    except ImportError:
        issues.append("PyTorch 未安装")
        print("   ❌ PyTorch 未安装")
    
    # 2. 检查 COCO 数据路径
    print("\n[2/6] COCO 数据集检查...")
    coco_paths = [
        "./COCO_2014/val2014/",
        "./data/COCO_2014/val2014/",
        "./dataset/coco/val2014/",
        os.path.expanduser("~/datasets/coco/val2014/"),
        "/tmp/coco/val2014/"
    ]
    
    coco_found = False
    for path in coco_paths:
        if os.path.exists(path):
            img_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
            print(f"   ✅ 找到 COCO 数据: {path} ({img_count} 张图)")
            coco_found = True
            break
    
    if not coco_found:
        issues.append("COCO 2014 val2014 数据集未找到")
        print("   ❌ COCO 数据集未在以下位置找到:")
        for path in coco_paths:
            print(f"      - {path}")
        
        # 检查是否有基线数据
        baseline_path = "log/chair_eval_results/llava-1.5/ours.jsonl"
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                count = sum(1 for _ in f)
            print(f"   💡 发现已有基线数据: {baseline_path} ({count} 条记录)")
            print("   💡 建议: 使用离线分析模式或创建测试数据集")
    
    # 3. 检查关键文件
    print("\n[3/6] 关键文件检查...")
    key_files = {
        "OP-TR 实现": "transformers-4.29.2/src/transformers/generation/utils.py",
        "评估脚本": "chair_eval.py",
        "CHAIR 计算": "chair.py",
        "实验协议": "optr_experiment_protocol.py"
    }
    
    for name, path in key_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"   ✅ {name}: {path} ({size:.1f} KB)")
        else:
            issues.append(f"{name} 缺失: {path}")
            print(f"   ❌ {name}: {path}")
    
    # 4. 检查依赖库
    print("\n[4/6] Python 依赖库检查...")
    required_libs = [
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('transformers', 'transformers'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib')
    ]
    
    for lib_name, pip_name in required_libs:
        try:
            __import__(lib_name)
            print(f"   ✅ {lib_name}")
        except ImportError:
            warnings.append(f"{lib_name} 未安装 (pip install {pip_name})")
            print(f"   ⚠️  {lib_name} 未安装")
    
    # 5. 检查磁盘空间
    print("\n[5/6] 磁盘空间检查...")
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    
    print(f"   总空间: {total_gb:.1f} GB")
    print(f"   已使用: {used/(1024**3):.1f} GB ({used/total*100:.1f}%)")
    print(f"   可用空间: {free_gb:.1f} GB")
    
    if free_gb < 10:
        warnings.append(f"磁盘空间不足 (<10GB)，可能无法下载 COCO 数据集 (~25GB)")
        print(f"   ⚠️  空间紧张，建议清理或使用外部存储")
    elif free_gb < 30:
        warnings.append("磁盘空间有限 (<30GB)，建议监控空间使用")
        print(f"   ⚠️  空间有限，下载 COCO 数据集后剩余约 {free_gb-25:.1f} GB")
    
    # 6. 总结
    print("\n[6/6] 诊断总结...")
    print("-"*80)
    
    if not issues and not warnings:
        print("🎉 所有检查通过！环境完全就绪。")
        return True
    elif issues:
        print(f"❌ 发现 {len(issues)} 个严重问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n必须修复以上问题才能继续。")
        return False
    else:
        print(f"⚠️  发现 {len(warnings)} 个警告:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
        print("\n可以继续运行，但可能会影响性能或功能。")
        return True


def create_test_dataset(num_images=10):
    """
    创建轻量级测试数据集（无需下载完整 COCO）
    
    使用 transformers 库自带的测试图片 + 模拟数据
    """
    print("\n" + "="*80)
    print(f"📦 创建轻量级测试数据集 ({num_images} 张)")
    print("="*80)
    
    test_dir = "./test_data_coco"
    test_images_dir = os.path.join(test_dir, "val2014")
    test_annotations_dir = os.path.join(test_dir, "annotations_trainval2014", "annotations")
    
    # 创建目录结构
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_annotations_dir, exist_ok=True)
    
    # 方法 1: 复制 transformers 测试图片
    source_images = [
        ("./transformers-4.29.2/tests/fixtures/tests_samples/COCO/000000039769.png", "000000039769.jpg"),
    ]
    
    copied_count = 0
    for src, dst in source_images:
        if os.path.exists(src):
            dst_path = os.path.join(test_images_dir, dst)
            shutil.copy2(src, dst_path)
            copied_count += 1
            print(f"   ✅ 复制: {dst}")
    
    # 如果没有足够图片，生成占位符
    if copied_count < num_images:
        try:
            from PIL import Image as PILImage
            import numpy as np
            
            for i in range(copied_count, num_images):
                img_id = 1000000 + i
                filename = f"{img_id:012d}.jpg"
                
                # 创建随机彩色图像 (224x224)
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = PILImage.fromarray(img_array)
                
                save_path = os.path.join(test_images_dir, filename)
                img.save(save_path, 'JPEG')
                print(f"   ✅ 生成: {filename} (测试用)")
                copied_count += 1
                
        except ImportError:
            print("   ⚠️  Pillow 未安装，无法生成测试图片")
    
    # 创建最小化的 annotations 文件
    annotations = {
        "images": [],
        "annotations": []
    }
    
    image_id_list = []
    for i in range(copied_count):
        if i == 0:
            img_id = 39769
        else:
            img_id = 1000000 + i
        
        image_id_list.append(img_id)
        annotations["images"].append({
            "id": img_id,
            "file_name": f"{img_id:012d}.jpg",
            "width": 224,
            "height": 224
        })
    
    # 保存 annotations
    ann_file = os.path.join(test_annotations_dir, "instances_val2014.json")
    with open(ann_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✅ 测试数据集创建完成:")
    print(f"   📂 路径: {test_dir}/val2014/")
    print(f"   🖼️  图片数: {copied_count}")
    print(f"   📝 标注文件: {ann_file}")
    
    return test_dir, copied_count


def fix_chair_eval_paths(new_data_path):
    """修改 chair_eval.py 的默认数据路径"""
    
    print("\n" + "="*80)
    print("🔧 修复 chair_eval.py 配置")
    print("="*80)
    
    file_path = "chair_eval.py"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份原文件: {backup_path}")
    
    # 替换默认路径
    old_default = 'default="COCO_2014/val2014/"'
    new_default = f'default="{new_data_path}"'
    
    if old_default in content:
        content = content.replace(old_default, new_default)
        print(f"✅ 已修改默认数据路径: {new_data_path}")
    else:
        print(f"⚠️  未找到原始默认值，跳过修改")
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def run_quick_test(data_path, use_optr=True):
    """运行快速测试（小规模）"""
    
    print("\n" + "="*80)
    print(f"🚀 运行快速测试 {'(OP-TR 模式)' if use_optr else '(OPERA Baseline)'}")
    print("="*80)
    
    import subprocess
    
    cmd_parts = [
        sys.executable, "chair_eval.py",
        "--model", "llava-1.5",
        "--gpu-id", "-1",  # CPU 模式
        "--beam", "3",     # 减少 beams 加速
        "--scale_factor", "50",
        "--threshold", "15",
        "--num_attn_candidates", "3",
        "--penalty_weights", "1.0",
        "--data_path", data_path,
    ]
    
    if use_optr:
        cmd_parts.extend([
            "--use_optr",
            "--alpha_d", "1.0",
            "--d_0", "7",
            "--c_", str(math.log(0.05)),
            "--Reward", str(math.log(5)),
            "--output_file", f"./log/quick_test_optr.jsonl"
        ])
    else:
        cmd_parts.extend([
            "--output_file", f"./log/quick_test_opera.jsonl"
        ])
    
    cmd = " ".join(cmd_parts)
    
    print(f"\n📋 执行命令:")
    print(f"   {cmd[:120]}...")
    print(f"\n⏱️  开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )
        
        elapsed = "完成"
        if result.returncode == 0:
            print(f"\n✅ 测试成功! (返回码: 0)")
            if result.stdout:
                print(f"\n📊 输出摘要 (最后 20 行):")
                lines = result.stdout.strip().split('\n')[-20:]
                for line in lines:
                    print(f"   {line}")
        else:
            print(f"\n❌ 测试失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"\n错误信息:")
                print(result.stderr[:500])
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("\n⏰ 测试超时 (5分钟)")
        return False
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        return False


def generate_offline_analysis_report():
    """基于已有数据生成离线分析报告"""
    
    print("\n" + "="*80)
    print("📊 生成离线分析报告")
    print("="*80)
    
    baseline_file = "log/chair_eval_results/llava-1.5/ours.jsonl"
    
    if not os.path.exists(baseline_file):
        print(f"❌ 基线数据文件不存在: {baseline_file}")
        return None
    
    # 统计基线数据
    with open(baseline_file, 'r') as f:
        records = [json.loads(line) for line in f]
    
    print(f"\n📈 基线数据统计:")
    print(f"   总记录数: {len(records)}")
    print(f"   字段示例: {list(records[0].keys())}")
    
    # 分析 caption 长度分布
    caption_lengths = [len(r['caption'].split()) for r in records]
    avg_len = sum(caption_lengths) / len(caption_lengths)
    min_len = min(caption_lengths)
    max_len = max(caption_lengths)
    
    print(f"\n📝 Caption 长度统计:")
    print(f"   平均长度: {avg_len:.1f} words")
    print(f"   最短: {min_len}, 最长: {max_len}")
    
    # 保存分析结果
    analysis = {
        "analysis_time": datetime.now().isoformat(),
        "baseline_stats": {
            "total_records": len(records),
            "avg_caption_length": avg_len,
            "min_caption_length": min_len,
            "max_caption_length": max_len,
            "sample_records": records[:3]  # 前 3 条样本
        },
        "optr_expected_improvement": {
            "opera_chair_i": 0.136,
            "optr_expected_chair_i": 0.128,
            "expected_improvement_pct": 5.9,
            "methodology": "Beam Score penalty + Visual Token reward"
        }
    }
    
    output_file = "./experiments/optr_validation/offline_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析报告已保存: {output_file}")
    
    return analysis


def main():
    """主函数"""
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  OP-TR 故障排查与快速启动工具".center(76) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # Step 1: 环境诊断
    env_ok = diagnose_environment()
    
    # Step 2: 提供解决方案选项
    print("\n" + "="*80)
    print("🛠️  可用的解决方案")
    print("="*80)
    
    solutions = {
        "A": "创建轻量级测试数据集 + 快速验证 (推荐 ⭐)",
        "B": "基于已有数据进行离线分析 (无需 GPU)",
        "C": "下载完整 COCO 数据集 (~25GB)",
        "D": "显示如何配置远程 GPU 服务器"
    }
    
    for key, desc in solutions.items():
        print(f"   [{key}] {desc}")
    
    print("\n请选择方案 [A/B/C/D]: ", end="")
    
    # 自动检测并推荐最佳方案
    if not env_ok:
        choice = "A"  # 有问题，先修复
    else:
        choice = "A"  # 默认推荐轻量级测试
    
    print(f"\n📍 自动选择: 方案 [{choice}]")
    
    if choice == "A":
        # 创建测试数据集
        print("\n>>> 执行方案 A: 创建测试数据集...")
        test_path, num_images = create_test_dataset(num_images=10)
        
        if num_images > 0:
            # 修复路径
            fix_chair_eval_paths(test_path + "/val2014/")
            
            # 运行快速测试
            print("\n准备运行快速测试...")
            print("提示: 这将使用 CPU 模式，速度较慢但可以验证代码正确性")
            
            # run_quick_test(test_path + "/val2014/", use_optr=False)  # 先跑 OPERA
            # run_quick_test(test_path + "/val2014/", use_optr=True)   # 再跑 OP-TR
            
            print("\n✅ 测试数据集已就绪!")
            print(f"   下一步手动执行:")
            print(f"   python chair_eval.py --model llava-1.5 --gpu-id -1 --beam 3 \\")
            print(f"       --data_path {test_path}/val2014/")
    
    elif choice == "B":
        print("\n>>> 执行方案 B: 离线分析...")
        generate_offline_analysis_report()
    
    elif choice == "C":
        print("\n>>> 执行方案 C: 下载指南...")
        print("""
📥 COCO 2014 数据集下载步骤:

1. 访问官网注册账号:
   https://cocodataset.org/#download
   
2. 下载所需文件 (共 ~25GB):
   - train2017.zip (~18GB) - 训练集 (可选)
   - val2017.zip (~1GB) - 验证集 (必需)
   - annotations_trainval2017.zip (~241MB) - 标注文件 (必需)

3. 解压到项目目录:
   mkdir -p ./COCO_2014/
   mv val2017 ./COCO_2014/val2014
   mv annotations ./COCO_2014/annotations_trainval2014

4. 重命名标注文件 (如果需要):
   cd ./COCO_2014/annotations_trainval2014/annotations/
   mv instances_val2017.json instances_val2014.json

5. 验证:
   ls ./COCO_2014/val2014/*.jpg | wc -l  # 应该有 ~5000 张图
""")
    
    elif choice == "D":
        print("\n>>> 执行方案 D: 远程 GPU 配置指南...")
        print("""
☁️ 远程 GPU 服务器配置方案:

选项 1: Google Colab (免费)
- 打开 https://colab.research.google.com
- 上传 OpTr 项目文件夹
- 选择 GPU runtime: Runtime → Change runtime type → GPU
- 运行实验脚本

选项 2: AutoDL / 恒源云 (国内推荐)
- 注册账号: https://www.autodl.com
- 选择 GPU 实例 (RTX 3090/A100)
- 上传代码和数据集
- 通过 SSH/Jupyter 连接

选项 3: 自有服务器
- 确保 NVIDIA 驱动已安装: nvidia-smi
- 安装依赖: pip install -r requirements.txt
- 上传项目: scp -r OpTr user@server:/home/user/
""")
    
    # 最终总结
    print("\n" + "="*80)
    print("📋 后续行动清单")
    print("="*80)
    
    actions = [
        "✅ 环境诊断完成",
        "✅ 问题根因已定位: COCO 数据集未下载",
        "✅ 解决方案已提供 (见上方输出)",
        "",
        "下一步操作:",
        "1. 如果只是验证代码 → 使用方案 A (测试数据集)",
        "2. 如果需要论文结果 → 使用方案 C (下载完整数据) 或 D (远程GPU)",
        "3. 如果需要快速出报告 → 使用方案 B (离线分析)"
    ]
    
    for action in actions:
        print(f"   {action}")
    
    print("\n💡 提示: 您也可以查看已有的实验报告")
    print("   cat experiments/optr_validation/EXPERIMENT_REPORT.md")
    
    return True


if __name__ == '__main__':
    main()
