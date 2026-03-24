"""
读取 PyTorch Lightning checkpoint 文件中的全部超参数并输出到命令行

用法:
    python -m src.utils.load_ckpt_hyperparams <checkpoint_path>

一些路径:
    checkpoints\vae-dnn-cvae\official\cvae_dense_fullgrid.ckpt
    checkpoints\vae-dnn-cvae\official\dnn_edge_fullgrid.ckpt
    checkpoints\vae-dnn-cvae\official\vae_incept_edges.ckpt
示例:
    python -m src.utils.load_ckpt_hyperparams checkpoints\vae-dnn-cvae\official\vae_incept_edges.ckpt
"""

import sys
import pprint
import torch


def load_checkpoint_hyperparams(ckpt_path: str):
    """加载 checkpoint 并打印所有超参数"""
    print(f"Loading checkpoint: {ckpt_path}\n")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print("=" * 60)
    print("Checkpoint keys:", list(checkpoint.keys()))
    print("=" * 60)

    # 1. LightningModule 通过 save_hyperparameters() 保存的超参数
    if 'hyper_parameters' in checkpoint:
        print("\n[1] hyper_parameters (LightningModule.save_hyperparameters):")
        print("-" * 60)
        hp = checkpoint['hyper_parameters']
        # pretty print
        for key, val in hp.items():
            print(f"  {key}: {val}")
    else:
        print("\n[1] hyper_parameters: NOT FOUND")

    # 2. 其他顶层元数据
    meta_keys = [k for k in checkpoint.keys() if k not in ('state_dict', 'hyper_parameters', 'optimizer_states')]
    if meta_keys:
        print(f"\n[2] Other metadata keys: {meta_keys}")
        for key in meta_keys:
            val = checkpoint[key]
            if not isinstance(val, (str, int, float, bool, type(None))):
                val = f"<{type(val).__name__}, len={len(val)}>"
            print(f"  {key}: {val}")

    # 3. state_dict 概览 (层名 + shape，不打印权重)
    if 'state_dict' in checkpoint:
        print(f"\n[3] state_dict ({len(checkpoint['state_dict'])} layers):")
        print("-" * 60)
        for key, val in checkpoint['state_dict'].items():
            print(f"  {key}: {list(val.shape)}, dtype={val.dtype}")

    # 4. 如果有嵌套的 vae 超参数（如 vae.latent_size 等）
    if 'hyper_parameters' in checkpoint:
        hp = checkpoint['hyper_parameters']
        if 'vae' in hp:
            print(f"\n[4] vae sub-model config from hyper_parameters:")
            print("-" * 60)
            vae_cfg = hp['vae']
            if hasattr(vae_cfg, '__dict__'):
                pprint.pprint(vae_cfg.__dict__)
            else:
                pprint.pprint(vae_cfg)

    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python -m src.utils.load_ckpt_hyperparams <checkpoint_path>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    load_checkpoint_hyperparams(ckpt_path)


if __name__ == '__main__':
    main()