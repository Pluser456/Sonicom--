"""
配置管理系统 - 基于 omegaconf
用法:
    from src.utils.config import load_config
    config = load_config("path/to/config.yaml")
    print(config.dataset.name)
    print(config.model.embed_dim)
"""
from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_path: str):
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        OmegaConf.DictConfig 对象，支持链式访问
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    cfg = OmegaConf.load(config_path)
    return cfg


def save_config(config, save_path: str):
    """
    保存配置到 YAML 文件

    Args:
        config: OmegaConf.DictConfig 对象
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(config, f)

if __name__ == "__main__":
    # 测试加载配置
    config = load_config(r"configs\vqvae\cnn-sub.yaml")
    print(config)
    save_config(config, "test_saved_config.yaml")
    