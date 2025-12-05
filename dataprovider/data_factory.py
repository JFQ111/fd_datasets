"""
数据加载器工厂函数 - 创建DataLoader并实现缓存机制
"""

import os
import pickle
import hashlib
import warnings
from pathlib import Path
from sympy import im
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torch

from .cwru_dataset import CWRUBearingDataset
from .pu_dataset import PUBearingDataset
from .jnu_dataset import JNUBearingDataset
from .mfpt_dataset import MFPTBearingDataset
from .hust_dataset import HUSTBearingDataset
from .hustv_dataset import HUSTVBearingDataset

warnings.filterwarnings('ignore')

class DomainLabeledDataset:
    def __init__(self, dataset, domain_name):
        self.dataset = dataset
        self.domain_name = domain_name
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            sample['domain'] = self.domain_name
        else:
            # 如果原始返回格式是tuple，转换为dict
            sample = {
                'data': sample[0] if len(sample) > 0 else None,
                'label': sample[1] if len(sample) > 1 else None,
                'domain': self.domain_name
            }
        return sample
    
    def __len__(self):
        return len(self.dataset)
def get_cache_key(args):
    """
    根据所有相关参数生成唯一的缓存键

    Args:
        args: 命令行参数对象

    Returns:
        str: 缓存键的哈希值
    """
    # 收集所有影响数据集的参数
    cache_params = {
        # 数据集选择参数
        'dataset': sorted(args.dataset) if isinstance(args.dataset, list) else args.dataset,
        'dataset_weights': args.dataset_weights,
        'root_path': args.root_path,

        # CWRU参数
        'data_source': args.data_source,
        'workloads': sorted(args.workloads) if hasattr(args, 'workloads') else None,
        'task_type': args.task_type,

        # PU参数
        'pu_workloads': sorted(args.pu_workloads) if hasattr(args, 'pu_workloads') else None,
        'pu_task_type': args.pu_task_type if hasattr(args, 'pu_task_type') else None,
        'pu_signal_type': args.pu_signal_type if hasattr(args, 'pu_signal_type') else None,

        # HUST参数
        'hust_workloads': sorted(args.hust_workloads) if hasattr(args, 'hust_workloads') else None,
        'hust_task_type': args.hust_task_type if hasattr(args, 'hust_task_type') else None,
        'hust_signal_type': args.hust_signal_type if hasattr(args, 'hust_signal_type') else None,

        # HUSTV参数
        'hustv_bearings': sorted(args.hustv_bearings) if hasattr(args, 'hustv_bearings') else None,
        'hustv_workloads': sorted(args.hustv_workloads) if hasattr(args, 'hustv_workloads') else None,
        'hustv_task_type': args.hustv_task_type if hasattr(args, 'hustv_task_type') else None,

        # JNU参数
        'jnu_workloads': sorted(args.jnu_workloads) if hasattr(args, 'jnu_workloads') else None,

        # 采样和窗口参数
        'sampling_rate': args.sampling_rate,
        'window_size': args.window_size,
        'stride': args.stride,

        # 数据集划分参数
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,

        # 预处理参数
        'normalize': args.normalize,
        'transform_type': args.transform_type,

        # 变换参数
        'cwt_scales': args.cwt_scales if hasattr(args, 'cwt_scales') else None,
        'stft_nperseg': args.stft_nperseg if hasattr(args, 'stft_nperseg') else None,
        'stft_noverlap': args.stft_noverlap if hasattr(args, 'stft_noverlap') else None,
        'gaf_method': args.gaf_method if hasattr(args, 'gaf_method') else None,
        'rp_eps': args.rp_eps if hasattr(args, 'rp_eps') else None,
        'target_size': args.target_size if hasattr(args, 'target_size') else None,
    }

    # 生成哈希键
    cache_str = str(sorted(cache_params.items()))
    cache_key = hashlib.md5(cache_str.encode()).hexdigest()

    return cache_key


def save_datasets_to_cache(datasets, cache_path):
    """
    保存数据集到缓存文件

    Args:
        datasets: 包含(train_dataset, val_dataset, test_dataset)的元组
        cache_path: 缓存文件路径
    """
    try:
        # 确保缓存目录存在
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving datasets to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cache saved successfully")

    except Exception as e:
        print(f"Warning: Failed to save cache: {str(e)}")


def load_datasets_from_cache(cache_path):
    """
    从缓存文件加载数据集

    Args:
        cache_path: 缓存文件路径

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) 或 None（如果加载失败）
    """
    try:
        if not cache_path.exists():
            return None

        print(f"Loading datasets from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            datasets = pickle.load(f)
        print(f"Cache loaded successfully")

        return datasets

    except Exception as e:
        print(f"Warning: Failed to load cache: {str(e)}")
        return None


def create_cached_datasets(args, cache_dir='./datasets/cache'):
    """
    创建数据集，支持缓存机制

    Args:
        args: 命令行参数对象
        cache_dir: 缓存目录路径

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, weighted_sampler)
    """
    # 数据集类映射
    DATASET_CLASSES = {
        'CWRU': CWRUBearingDataset,
        'JNU': JNUBearingDataset,
        'PU': PUBearingDataset,
        'MFPT': MFPTBearingDataset,
        'HUST': HUSTBearingDataset,
        'HUSTV': HUSTVBearingDataset
    }

    # 处理数据集参数
    if isinstance(args.dataset, str):
        datasets_to_use = [args.dataset]
    elif isinstance(args.dataset, list):
        datasets_to_use = args.dataset
    else:
        raise ValueError(f"args.dataset must be str or list, got {type(args.dataset)}")

    # 验证数据集类型
    for dataset_name in datasets_to_use:
        if dataset_name not in DATASET_CLASSES:
            raise ValueError(f"Unknown dataset type: {dataset_name}. Available: {list(DATASET_CLASSES.keys())}")

    # 生成缓存键和缓存路径
    cache_key = get_cache_key(args)
    cache_dir_path = Path(cache_dir)
    cache_file = cache_dir_path / f"datasets_{cache_key}.pkl"

    print(f"Cache key: {cache_key}")
    print(f"Cache file: {cache_file}")

    # 尝试从缓存加载
    cached_datasets = load_datasets_from_cache(cache_file)

    if cached_datasets is not None:
        train_dataset, val_dataset, test_dataset = cached_datasets
        print(f"Loaded datasets from cache:")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")
    else:
        # 缓存不存在，创建新数据集
        print(f"Cache not found, creating new datasets from: {datasets_to_use}")

        train_datasets = []
        val_datasets = []
        test_datasets = []

        # 为每个数据集创建训练、验证、测试集
        for dataset_name in datasets_to_use:
            print(f"  Loading {dataset_name} dataset...")

            dataset_class = DATASET_CLASSES[dataset_name]

            try:
                train_ds = dataset_class(args, flag='train')
                val_ds = dataset_class(args, flag='val')
                test_ds = dataset_class(args, flag='test')

                # 为每个数据集中的样本添加域标签
                # 使用包装器添加域标签
                train_ds = DomainLabeledDataset(train_ds, dataset_name)
                val_ds = DomainLabeledDataset(val_ds, dataset_name)
                test_ds = DomainLabeledDataset(test_ds, dataset_name)
                train_datasets.append(train_ds)
                val_datasets.append(val_ds)
                test_datasets.append(test_ds)

                print(f"    - Training: {len(train_ds)} samples")
                print(f"    - Validation: {len(val_ds)} samples")
                print(f"    - Test: {len(test_ds)} samples")

            except Exception as e:
                print(f"    Error loading {dataset_name}: {str(e)}")
                raise

        # 组合多个数据集
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
            val_dataset = val_datasets[0]
            test_dataset = test_datasets[0]
        else:
            print("  Combining multiple datasets...")
            train_dataset = ConcatDataset(train_datasets)
            val_dataset = ConcatDataset(val_datasets)
            test_dataset = ConcatDataset(test_datasets)
        # 保存到缓存
        datasets_to_cache = (train_dataset, val_dataset, test_dataset)
        save_datasets_to_cache(datasets_to_cache, cache_file)

        print(f"\nDataset creation completed:")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")

    # 创建加权采样器（如果需要）
    weighted_sampler = None
    if hasattr(args, 'dataset_weights') and args.dataset_weights is not None and args.dataset_weights != 'None':
        print("Creating weighted sampler for training...")

        # 解析权重字符串
        weights_dict = {}
        try:
            for weight_pair in args.dataset_weights.split(','):
                dataset_name, weight = weight_pair.split(':')
                weights_dict[dataset_name.strip()] = float(weight.strip())
            print(f"  Dataset weights: {weights_dict}")
        except ValueError as e:
            print(f"  Warning: Invalid weight format '{args.dataset_weights}', using equal weights")
            weights_dict = {name: 1.0 for name in datasets_to_use}

        # 为训练集中的每个样本分配权重
        sample_weights = []

        if len(datasets_to_use) == 1:
            # 单个数据集情况
            dataset_name = datasets_to_use[0]
            weight = weights_dict.get(dataset_name, 1.0)
            sample_weights = [weight] * len(train_dataset)
            print(f"  Applied weight {weight} to {len(train_dataset)} samples from {dataset_name}")
        else:
            if cached_datasets is not None:
                if isinstance(train_dataset, ConcatDataset):
                    sub_datasets = train_dataset.datasets
                else:
                    sub_datasets = [train_dataset]  # fallback: 单个数据集

                if len(sub_datasets) != len(datasets_to_use):
                    print("Warning: Cached sub-dataset count mismatch, falling back to equal weights")
                    total_size = len(train_dataset)
                    default_weight = 1.0
                    sample_weights = [default_weight] * total_size
                    print(f"  Applied fallback weight {default_weight} to {total_size} samples")
                else:
                    for i, dataset_name in enumerate(datasets_to_use):
                        weight = weights_dict.get(dataset_name, 1.0)
                        dataset_size = len(sub_datasets[i])
                        sample_weights.extend([weight] * dataset_size)
                        print(f"  Applied weight {weight} to {dataset_size} samples from {dataset_name}")

            else:
                # 新创建的情况，直接使用已有的train_datasets
                for i, dataset_name in enumerate(datasets_to_use):
                    weight = weights_dict.get(dataset_name, 1.0)
                    dataset_size = len(train_datasets[i])
                    sample_weights.extend([weight] * dataset_size)
                    print(f"  Applied weight {weight} to {dataset_size} samples from {dataset_name}")
        # 创建加权随机采样器
        weighted_sampler = WeightedRandomSampler(
            weights=torch.FloatTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"  Created WeightedRandomSampler for {len(sample_weights)} samples")

    print(f"  - Label mapping: Normal=0, Inner_Race=1, Outer_Race=2, Ball=3")

    return train_dataset, val_dataset, test_dataset, weighted_sampler


def create_dataloaders(args, cache_dir='./datasets/cache'):
    """
    创建数据加载器，包含缓存机制

    Args:
        args: 命令行参数对象，需要包含以下属性：
            - batch_size: 批次大小
            - num_workers: 数据加载进程数
            - pin_memory: 是否固定内存
            - drop_last: 是否丢弃最后不完整的批次
            - shuffle: 是否打乱数据（仅在没有weighted_sampler时生效）
        cache_dir: 缓存目录路径

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 设置默认参数
    batch_size = getattr(args, 'batch_size', 32)
    num_workers = getattr(args, 'num_workers', 4)
    pin_memory = getattr(args, 'pin_memory', True)
    drop_last = getattr(args, 'drop_last', False)
    shuffle = getattr(args, 'shuffle', True)

    print(f"Creating DataLoaders with:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - Pin memory: {pin_memory}")
    print(f"  - Drop last: {drop_last}")
    print(f"  - Shuffle: {shuffle}")

    # 创建数据集
    train_dataset, val_dataset, test_dataset, weighted_sampler = create_cached_datasets(args, cache_dir)

    # 创建训练集DataLoader
    if weighted_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=weighted_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        print("Using WeightedRandomSampler for training data")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

    # 创建验证集和测试集DataLoader（通常不需要打乱）
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"\nDataLoaders created successfully:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def clear_cache(cache_dir='./datasets/cache', pattern=None):
    """
    清理缓存文件

    Args:
        cache_dir: 缓存目录路径
        pattern: 文件名模式（可选），如果为None则清理所有缓存文件
    """
    cache_dir_path = Path(cache_dir)

    if not cache_dir_path.exists():
        print(f"Cache directory does not exist: {cache_dir_path}")
        return

    if pattern:
        cache_files = list(cache_dir_path.glob(pattern))
    else:
        cache_files = list(cache_dir_path.glob("datasets_*.pkl"))

    if not cache_files:
        print("No cache files found to clear")
        return

    print(f"Clearing {len(cache_files)} cache files...")
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            print(f"  Deleted: {cache_file.name}")
        except Exception as e:
            print(f"  Failed to delete {cache_file.name}: {str(e)}")

    print("Cache cleanup completed")
