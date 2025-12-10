'''
author: Your Name
date: 2025-8-11
description: HUST(V)轴承数据集加载器范例
'''
from dataprovider import create_dataloaders
import argparse
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
    parser = argparse.ArgumentParser(description='HUST(V) Bearing Dataset Example')
    
    ## 数据集参数
    # 数据集选择参数
    parser.add_argument('--dataset',
                        type=str,
                        nargs='+',
                        default=['HUSTV'],
                        choices=['CWRU', 'PU', 'JNU', 'MFPT', 'HUST', 'HUSTV'],
                        help='数据集类型选择')

    # 数据集权重参数（可选）
    parser.add_argument('--dataset_weights',
                        type=str,
                        default=None,
                        help='各数据集权重，格式: "HUSTV:1.0"')

    # 数据集基本参数
    parser.add_argument('--root_path', type=str, default='./datasets',
                        help='数据集根目录')

    # HUST(V)数据集参数
    parser.add_argument('--hustv_bearings', type=str, nargs='+', 
                        default=['6204', '6205', '6206', '6207', '6208'],
                        choices=['6204', '6205', '6206', '6207', '6208'],
                        help='HUST(V)轴承型号选择，可多选: 6204 6205 6206 6207 6208')
    
    parser.add_argument('--hustv_workloads', type=str, nargs='+', 
                        default=['0', '2', '4'],
                        choices=['0', '2', '4'],
                        help='HUST(V)工况选择，可多选: 0(0W) 2(200W) 4(400W)')
    
    parser.add_argument('--hustv_task_type', type=str, default='4class',
                        choices=['4class', '7class'],
                        help='HUST(V)分类任务类型: 4class(基础故障) 或 7class(包含复合故障)')

    # CWRU数据集参数
    parser.add_argument('--data_source', type=str, default='12k_DE',
                        choices=['12k_DE', '48k_DE', 'both'],
                        help='CWRU数据源选择: 12k_DE, 48k_DE, 或 both(使用两种采样率)')
    parser.add_argument('--workloads', type=str, nargs='+', default=['0hp', '1hp', '2hp', '3hp'],
                        help='CWRU工况选择，可多选: 0hp 1hp 2hp 3hp')
    parser.add_argument('--task_type', type=str, default='4class',
                        choices=['4class', '10class'],
                        help='CWRU分类任务类型: 4class(正常,内圈,外圈,滚动体) 或 10class(细分故障)')
    # PU数据集参数
    parser.add_argument('--pu_workloads', type=str, nargs='+',
                        default=['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04'],
                        help='PU工况选择，可多选')
    parser.add_argument('--pu_task_type', type=str, default='3class_artificial',
                        choices=['3class_artificial', '5class_artificial', '9class_artificial',
                                 '13class_artificial', '15class_nature'],
                        help='PU分类任务类型')
    parser.add_argument('--pu_signal_type', type=str, default='vibration',
                        choices=['vibration', 'current', 'both'],
                        help='PU信号类型选择: vibration(振动), current(电流), both(两者)')

    # JNU数据集参数
    parser.add_argument('--jnu_workloads', type=str, nargs='+', default=['600', '800', '1000'],
                        help='JNU工况选择，可多选，支持转速: 600, 800, 1000 等')

    # 采样率和窗口参数
    parser.add_argument('--sampling_rate', type=int, default=50000,
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
    parser.add_argument('--cwt_scales', type=int, default=64,
                        help='CWT尺度数量')
    parser.add_argument('--stft_nperseg', type=int, default=64,
                        help='STFT窗口长度')
    parser.add_argument('--stft_noverlap', type=int, default=32,
                        help='STFT重叠长度')
    parser.add_argument('--gaf_method', type=str, default='summation',
                        choices=['summation', 'difference'],
                        help='GAF方法: summation(GASF) 或 difference(GADF)')
    parser.add_argument('--rp_eps', type=float, default=0.1,
                        help='递归图阈值')
    parser.add_argument('--target_size', nargs=2, type=int, default=(32, 32),
                        help='目标图像大小 (宽度和高度)')

    ## dataloader参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='是否使用pin_memory')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='是否打乱数据集')
    parser.add_argument('--drop_last', type=bool, default=True,
                        help='是否丢弃最后一个不完整批次')

    args = parser.parse_args()
    
    # 转换参数格式
    args.target_size = tuple(args.target_size)
    
    print("=" * 60)
    print("HUST(V) Bearing Dataset Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Bearings: {args.hustv_bearings}")
    print(f"Workloads: {args.hustv_workloads}")
    print(f"Task type: {args.hustv_task_type}")
    print(f"Sampling rate: {args.sampling_rate}Hz")
    print(f"Window size: {args.window_size}")
    print(f"Transform type: {args.transform_type}")
    print(f"Normalize: {args.normalize}")
    print("=" * 60)
    
    # 创建数据集
    try:
        train_loader, val_loader, test_loader = create_dataloaders(args)
        
        print("\nDataset loaded successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # 测试数据加载
        print("\nTesting data loading...")
        for batch_idx, batch in enumerate(train_loader):
            data, labels, domain = unpack_batch(batch)
            domain_info = format_domain(domain)
            print(f"Batch {batch_idx}:")
            print(f"  Data shape: {data.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Label range: {labels.min().item()} - {labels.max().item()}")
            print(f"  Unique labels: {torch.unique(labels).tolist()}")
            print(f"  Domain(s): {domain_info}")
            
            if batch_idx >= 3:  # 只测试几个batch
                break
                
        print("\nClass Information:")
        if args.hustv_task_type == '4class':
            print("4-Class Task:")
            print("  0: Normal")
            print("  1: Inner Race Fault")
            print("  2: Outer Race Fault") 
            print("  3: Ball Fault")
        else:
            print("7-Class Task:")
            print("  0: Normal")
            print("  1: Inner Race Fault")
            print("  2: Outer Race Fault")
            print("  3: Ball Fault")
            print("  4: Inner-Outer Compound Fault")
            print("  5: Inner-Ball Compound Fault")
            print("  6: Outer-Ball Compound Fault")
            
        print(f"\nNote: Bearing 6204 missing Ball(B) and Inner-Ball(IB) faults")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
