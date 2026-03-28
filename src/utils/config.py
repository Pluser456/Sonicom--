"""
配置管理系统 - 统一管理数据集、模型、训练超参数
"""
import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace


def dict_to_namespace(d):
    """将嵌套字典转换为支持链式访问的对象"""
    if d is None:
        return None
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    ear_dir: str
    hrtf_dir: str
    input_form: str
    use_diff: bool
    mode: str
    n_folds: int
    val_fold: int
    seed: int = 42

@dataclass
class ModelConfig:
    """模型配置"""
    name: str


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int
    epochs: int
    learning_rate: float
    device: str
    log: bool = True
    continue_exp: str = None  # 继续的实验文件夹，如 "exp_001"


@dataclass
class PathsConfig:
    """路径配置"""
    data_dir: str
    log_dir: str
    checkpoint_dir: str


@dataclass
class Config:
    """完整配置"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(config_path: str) -> Config:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Config 对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    # 获取 dataclass 定义的字段名
    dataset_fields = {f.name for f in fields(DatasetConfig)}
    model_fields = {f.name for f in fields(ModelConfig)}
    training_fields = {f.name for f in fields(TrainingConfig)}
    paths_fields = {f.name for f in fields(PathsConfig)}

    # 分离基础字段和额外字段
    dataset_base = {k: v for k, v in data.get('dataset', {}).items() if k in dataset_fields}
    dataset_extra = {k: v for k, v in data.get('dataset', {}).items() if k not in dataset_fields}

    model_base = {k: v for k, v in data.get('model', {}).items() if k in model_fields}
    model_extra = {k: v for k, v in data.get('model', {}).items() if k not in model_fields}

    training_base = {k: v for k, v in data.get('training', {}).items() if k in training_fields}
    training_extra = {k: v for k, v in data.get('training', {}).items() if k not in training_fields}

    paths_base = {k: v for k, v in data.get('paths', {}).items() if k in paths_fields}
    paths_extra = {k: v for k, v in data.get('paths', {}).items() if k not in paths_fields}

    # 创建配置对象
    config = Config(
        dataset=DatasetConfig(**dataset_base),
        model=ModelConfig(**model_base),
        training=TrainingConfig(**training_base),
        paths=PathsConfig(**paths_base)
    )

    # 动态添加额外字段（嵌套字典转换为可访问对象）
    for key, value in dataset_extra.items():
        setattr(config.dataset, key, dict_to_namespace(value))
    for key, value in model_extra.items():
        setattr(config.model, key, dict_to_namespace(value))
    for key, value in training_extra.items():
        setattr(config.training, key, dict_to_namespace(value))
    for key, value in paths_extra.items():
        setattr(config.paths, key, dict_to_namespace(value))

    return config


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()
