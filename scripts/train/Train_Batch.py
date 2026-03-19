"""
批量训练脚本 - 自动连续训练多个配置
用法:
    python Train_Batch.py --script Train_VQVAE.py --configs config1.yaml config2.yaml
    python Train_Batch.py --script Train_PRTFNet.py --configs config1.yaml config2.yaml config3.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Training')
    parser.add_argument('--script', type=str, required=True,
                        help='Training script to run (e.g., Train_VQVAE.py, Train_PRTFNet.py)')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                        help='List of config files to train sequentially')
    parser.add_argument('--weightnames', type=str, nargs='+', default=None,
                        help='Optional: weight names for each config. Use "_" to skip loading pretrained weights (will use generated weightname)')
    return parser.parse_args()

def main():
    args = parse_args()

    # 如果没有指定权重名称，自动生成
    if args.weightnames is None:
        args.weightnames = [f"exp_{i}" for i in range(len(args.configs))]

    # 获取训练脚本的路径
    train_script = Path(__file__).parent / args.script

    for i, (config, weightname) in enumerate(zip(args.configs, args.weightnames)):
        print(f"\n{'='*60}")
        print(f"Starting training {i+1}/{len(args.configs)}: {config}")
        if weightname != "_":
            print(f"Weight name: {weightname}")
        print(f"{'='*60}\n")

        # 构建命令
        cmd = [
            sys.executable,
            str(train_script),
            "--config", config
        ]

        # 如果权重名不是下划线，才传递 --weightname 参数
        if weightname != "_":
            cmd.extend(["--weightname", weightname])

        # 执行训练 - 实时显示输出（tqdm进度条等）
        # 不使用 capture_output，确保输出直接显示在命令行
        result = subprocess.run(cmd, stdout=None, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"Training {config} failed with exit code {result.returncode}")
            # 可以选择 continue 继续训练下一个
            continue

        print(f"\nTraining {config} completed successfully!")

    print(f"\n{'='*60}")
    print("All training jobs completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
