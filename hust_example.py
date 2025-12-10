'''
author: Your Name
date: 2025-8-10
description: HUST轴承数据集加载器使用范例
'''
from dataprovider import create_dataloaders
import argparse
import numpy as np
import torch


def unpack_batch(batch):
    """Handle both dict (with domain) and tuple batch formats."""
    if isinstance(batch, dict):
        return batch.get('data'), batch.get('label'), batch.get('domain')
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1], None
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def format_domain(domain):
    """Format domain field for readable logging."""
    if domain is None:
        return 'N/A'
    if torch.is_tensor(domain):
        values = domain.detach().view(-1).tolist()
        unique_domains = sorted(set(str(v) for v in values))
    elif isinstance(domain, (list, tuple)):
        unique_domains = sorted(set(str(v) for v in domain))
    else:
        unique_domains = [str(domain)]
    return ', '.join(unique_domains)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ## 数据集参数
    # 数据集选择参数
    parser.add_argument('--dataset', type=str, nargs='+', default=['HUST'],
                        choices=['CWRU', 'PU', 'JNU', 'MFPT', 'HUST'],
                        help='数据集类型选择')
                        
    # 数据集权重参数（可选）
    parser.add_argument('--dataset_weights', type=str, default=None,
                        help='各数据集权重，格式: "CWRU:1.0,HUST:0.8"')

    # 数据集基本参数
    parser.add_argument('--root_path', type=str, default='./datasets',
                        help='数据集根目录')

    # HUST数据集参数
    parser.add_argument('--hust_workloads', type=str, nargs='+', 
                        default=['20Hz', '25Hz', '30Hz'],
                        help='HUST工况选择，可多选。支持具体工况如 20Hz 25Hz 或 all(所有工况)')
    parser.add_argument('--hust_task_type', type=str, default='4class',
                        choices=['4class', '9class'],
                        help='HUST分类任务类型: 4class(正常,内圈,外圈,滚动体) 或 9class(细分故障程度)')
    parser.add_argument('--hust_signal_type', type=str, default='x',
                        choices=['x', 'y', 'z', 'xyz'],
                        help='HUST信号类型选择: x(x方向), y(y方向), z(z方向), xyz(三方向组合)')

    # CWRU数据集参数（如果需要混合使用）
    parser.add_argument('--data_source', type=str, default='12k_DE',
                        choices=['12k_DE', '48k_DE', 'both'])
    parser.add_argument('--workloads', type=str, nargs='+', default=['0hp'],
                        help='CWRU工况选择')
    parser.add_argument('--task_type', type=str, default='4class',
                        choices=['4class', '10class'])

    # 其他数据集参数（保持兼容性）
    parser.add_argument('--pu_workloads', type=str, nargs='+',
                        default=['N15_M07_F10'])
    parser.add_argument('--pu_task_type', type=str, default='3class_artificial')
    parser.add_argument('--pu_signal_type', type=str, default='vibration')
    parser.add_argument('--jnu_workloads', type=str, nargs='+', default=['600'])

    # 采样率和窗口参数
    parser.add_argument('--sampling_rate', type=int, default=10000,
                        help='采样率 (Hz)')
    parser.add_argument('--window_size', type=int, default=1024,
                        help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=512,
                        help='滑动窗口步长')

    # 数据集划分参数
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')

    # 数据预处理参数
    parser.add_argument('--normalize', default=True,
                        help='是否标准化')
    parser.add_argument('--transform_type', type=str, default='None',
                        choices=['None', 'cwt', 'stft', 'gaf', 'rp', 'scalogram'],
                        help='数据变换类型')

    # 变换参数
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--stft_nperseg', type=int, default=64)
    parser.add_argument('--stft_noverlap', type=int, default=32)
    parser.add_argument('--gaf_method', type=str, default='summation')
    parser.add_argument('--rp_eps', type=float, default=0.1)
    parser.add_argument('--target_size', nargs=2, type=int, default=(32, 32))

    ## dataloader参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--drop_last', type=bool, default=True)

    args = parser.parse_args()
    
    # 转换参数
    args.target_size = tuple(args.target_size)
    
    print("=== HUST Bearing Dataset Loading Example ===")
    print(f"Dataset: {args.dataset}")
    print(f"HUST Task Type: {args.hust_task_type}")
    print(f"HUST Workloads: {args.hust_workloads}")
    print(f"HUST Signal Type: {args.hust_signal_type}")
    print(f"Window Size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Sampling Rate: {args.sampling_rate}Hz")
    print(f"Transform Type: {args.transform_type}")
    print()

    # 创建数据集
    try:
        train_loader, val_loader, test_loader = create_dataloaders(args)
        
        print("=== Dataset Information ===")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print()

        # 测试数据加载
        print("=== Testing Data Loading ===")
        for batch_idx, batch in enumerate(train_loader):
            data, labels, domain = unpack_batch(batch)
            domain_info = format_domain(domain)
            print(f"Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}, domain(s) = {domain_info}")
            print(f"Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"Unique labels in batch: {sorted(labels.unique().tolist())}")
            
            if batch_idx >= 2:  # 只测试几个batch
                break
                
        print("\n=== Sample Data Statistics ===")
        sample_batch = next(iter(train_loader))
        sample_batch_data, sample_batch_labels, sample_batch_domain = unpack_batch(sample_batch)
        print(f"Sample data - Min: {sample_batch_data.min():.6f}, Max: {sample_batch_data.max():.6f}")
        print(f"Sample data - Mean: {sample_batch_data.mean():.6f}, Std: {sample_batch_data.std():.6f}")
        print(f"Sample domain(s): {format_domain(sample_batch_domain)}")
        
        # 打印类别分布
        all_labels = []
        for batch in train_loader:
            _, labels, _ = unpack_batch(batch)
            all_labels.extend(labels.tolist())
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"\nClass distribution in training set:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check:")
        print("1. HUST dataset files are in ./datasets/HUST/ directory")
        print("2. Files are in Excel format (.xlsx or .xls)")
        print("3. File naming follows the pattern like '0.5X_B_20Hz.xlsx'")
        print("4. Files contain data starting from row 23 with xyz acceleration in columns 3,4,5")
