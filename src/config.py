"""
统一配置管理
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_PATHS = {
    "sonicom": {
        "hrtf": PROJECT_ROOT / "data" / "sonicom" / "hrtf",
        "ear_voxel": PROJECT_ROOT / "data" / "sonicom" / "ear_voxel",
        "ear_image": PROJECT_ROOT / "data" / "sonicom" / "ear_image",
    },
    "widedspread": {
        "hrtf": PROJECT_ROOT / "data" / "widedspread" / "hrtf",
        "ear_voxel": PROJECT_ROOT / "data" / "widedspread" / "ear_voxel",
        "ear_image": PROJECT_ROOT / "data" / "widedspread" / "ear_image",
    },
}

# 模型默认配置
MODEL_CONFIG = {
    "vqvae": {
        "embed_dim": 192,
        "num_codebook_embeddings": 6,
        "commitment_cost_beta": 0.25,
        "decay": 0.99,
    },
    "cnn": {
        "in_channels": 1,
        "hidden_dims": [64, 128, 256],
    },
    "vae": {
        "latent_dim": 128,
    },
}

# 训练默认配置
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "device": "cuda",
    "save_interval": 10,
}

# 数据集默认配置
DATASET_CONFIG = {
    "sonicom": {
        "num_positions": 2562,
        "input_form": "voxel",  # "voxel" or "image"
        "use_diff": True,
    },
    "widedspread": {
        "num_positions": 2562,
        "input_form": "voxel",
        "use_diff": True,
    },
}


def get_data_path(dataset: str, data_type: str) -> Path:
    """获取数据路径"""
    return DATA_PATHS[dataset][data_type]


def get_model_config(model_name: str) -> dict:
    """获取模型配置"""
    return MODEL_CONFIG.get(model_name, {})


def get_training_config() -> dict:
    """获取训练配置"""
    return TRAINING_CONFIG.copy()


def get_dataset_config(dataset: str) -> dict:
    """获取数据集配置"""
    return DATASET_CONFIG.get(dataset, {})
