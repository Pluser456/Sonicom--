"""
批量评估脚本 - 自动连续评估多个配置
用法:
    python Eval_Batch.py --script eval_lsd.py --configs config1.yaml config2.yaml
    python Eval_Batch.py --script gen_hrtf_cnn.py --configs config1.yaml config2.yaml config3.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Evaluation')
    parser.add_argument('--script', type=str, required=True,
                        help='Evaluation script to run (e.g., eval_lsd.py, gen_hrtf_cnn.py)')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                        help='List of config files to evaluate sequentially')
    return parser.parse_args()

def main():
    args = parse_args()

    # 获取评估脚本的路径
    eval_script = Path(__file__).parent / args.script

    for i, config in enumerate(args.configs):
        print(f"\n{'='*60}")
        print(f"Starting evaluation {i+1}/{len(args.configs)}: {config}")
        print(f"{'='*60}\n")

        # 构建命令
        cmd = [
            sys.executable,
            str(eval_script),
            "--config", config
        ]

        # 执行评估 - 实时显示输出（tqdm进度条等）
        # 不使用 capture_output，确保输出直接显示在命令行
        result = subprocess.run(cmd, stdout=None, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"Evaluation {config} failed with exit code {result.returncode}")
            # 可以选择 continue 继续评估下一个
            continue

        print(f"\nEvaluation {config} completed successfully!")

    print(f"\n{'='*60}")
    print("All evaluation jobs completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
