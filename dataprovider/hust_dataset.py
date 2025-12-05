import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class HUSTBearingDataset(Dataset):
    """
    HUST Bearing Fault Diagnosis Dataset
    支持4分类和9分类任务，支持多工况组合
    数据来源：Excel文件，包含xyz三个方向的加速度信号
    """

    def __init__(self, args, flag='train'):
        """
        Args:
            args: 命令行参数对象
            flag: 'train', 'val', 'test'
        """
        self.args = args
        self.flag = flag
        
        # 默认采样率
        if not hasattr(args, 'sampling_rate') or args.sampling_rate is None:
            self.args.sampling_rate = 10000

        # 初始化标准化器
        if args.normalize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # 定义故障模式映射 - 根据您的数据调整
        self.fault_patterns = {
            # 正常状态
            'H': {'class_9': 0, 'class_4': 0, 'name': 'healthy'},
            
            # 内圈故障
            'I': {'class_9': 1, 'class_4': 1, 'name': 'inner_light'},
            '0.5X_I': {'class_9': 2, 'class_4': 1, 'name': 'inner_medium'},
            'X_I': {'class_9': 3, 'class_4': 1, 'name': 'inner_severe'},
            
            # 外圈故障
            'O': {'class_9': 4, 'class_4': 2, 'name': 'outer_light'},
            '0.5X_O': {'class_9': 5, 'class_4': 2, 'name': 'outer_medium'},
            'X_O': {'class_9': 6, 'class_4': 2, 'name': 'outer_severe'},
            
            # 滚动体故障
            'B': {'class_9': 7, 'class_4': 3, 'name': 'ball_light'},
            '0.5X_B': {'class_9': 8, 'class_4': 3, 'name': 'ball_medium'},
            
            # 复合故障 - 在4分类中忽略
            'C': {'class_9': 9, 'class_4': -1, 'name': 'combination_light'},
            '0.5X_C': {'class_9': 10, 'class_4': -1, 'name': 'combination_medium'},
        }

        # 定义类别名称
        self.class_names_9 = ['healthy', 'inner_light', 'inner_medium', 'inner_severe',
                              'outer_light', 'outer_medium', 'outer_severe',
                              'ball_light', 'ball_medium', 'combination_light', 'combination_medium']

        self.class_names_4 = ['healthy', 'inner', 'outer', 'ball']

        # 设置当前任务的类别名称
        if hasattr(args, 'hust_task_type') and args.hust_task_type == '4class':
            self.class_names = self.class_names_4
            self.num_classes = 4
        else:  # 9class或更多
            self.class_names = self.class_names_9
            self.num_classes = len(self.class_names_9)

        # 加载数据
        self._load_data()
        
        # 如果需要标准化，则对所有数据进行标准化
        if self.args.normalize and self.scaler is not None:
            self.data = self.scaler.transform(self.data)

        # 如果有变换类型，则提前应用变换
        if hasattr(self.args, 'transform_type') and self.args.transform_type != 'None':
            self.data = self._apply_transform_to_all_data()

    def _apply_transform_to_all_data(self):
        transformed_data = []
        for signal_window in self.data:
            transformed_signal = apply_transform(signal_window, self.args.sampling_rate, self.args)
            if len(transformed_signal.shape) == 2:
                transformed_signal = np.expand_dims(transformed_signal, axis=0)
            transformed_data.append(transformed_signal)
        return np.array(transformed_data)

    def _identify_fault_type(self, filename):
        """
        根据文件名识别故障类型
        """
        basename = filename.replace('.xlsx', '').replace('.xls', '')
        
        # 按长度降序排列模式，避免短模式匹配长模式
        patterns = ['0.5X_I', '0.5X_O', '0.5X_B', '0.5X_C', 'X_I', 'X_O', 'X_B', 'X_C', 'I', 'O', 'B', 'C', 'H']
        
        for pattern in patterns:
            if basename.startswith(pattern + '_') or basename == pattern:
                return pattern
                
        return None

    def _extract_workload(self, filename):
        """
        从文件名中提取工况信息 - 更灵活的匹配
        """
        import re
        
        # 使用正则表达式匹配所有数字+Hz的模式
        hz_match = re.search(r'(\d+(?:\.\d+)?)Hz', filename, re.IGNORECASE)
        if hz_match:
            return hz_match.group(0)
        
        # 匹配变速工况
        if 'VS_' in filename:
            return 'VS'
            
        return None

    def _get_workload_files(self, data_path, workload):
        """
        获取指定工况下的所有Excel文件
        """
        files_info = []
        
        if not os.path.exists(data_path):
            print(f"Warning: HUST data path '{data_path}' does not exist.")
            return files_info

        # 获取所有Excel文件
        excel_files = []
        for file in os.listdir(data_path):
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(file)

        excel_files.sort()
        
        print(f"Found {len(excel_files)} Excel files in {data_path}")

        for excel_file in excel_files:
            file_path = os.path.join(data_path, excel_file)
            
            # 识别故障类型
            fault_type = self._identify_fault_type(excel_file)
            if fault_type is None:
                print(f"Warning: Unknown fault pattern in file '{excel_file}', skipping...")
                continue
                
            # 提取工况
            file_workload = self._extract_workload(excel_file)
            if file_workload is None:
                print(f"Warning: Unknown workload in file '{excel_file}', skipping...")
                continue
                
            # 根据工况筛选文件
            if workload != 'all' and file_workload != workload:
                continue
                
            files_info.append((file_path, fault_type, file_workload))
            print(f"Loaded: {excel_file} -> {fault_type} -> {file_workload}")

        return files_info

    def _read_excel_with_multiple_engines(self, file_path):
        """
        尝试使用多个引擎读取Excel文件
        """
        engines = ['openpyxl', 'xlrd', None]  # None让pandas自动选择
        
        for engine in engines:
            try:
                if engine is None:
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_excel(file_path, engine=engine)
                return df
            except Exception as e:
                print(f"Engine {engine} failed for {file_path}: {e}")
                continue
                
        raise Exception(f"All engines failed to read {file_path}")

    def _clean_signal_data(self, signal_data):
        """
        清理信号数据，处理非数值类型和NaN值
        """
        # 首先尝试转换为数值类型
        try:
            # 如果是字符串类型的数组，尝试转换为浮点数
            if signal_data.dtype == 'object' or signal_data.dtype.kind in ['U', 'S']:
                # 替换可能的非数值字符串
                signal_data = pd.to_numeric(signal_data, errors='coerce')
            
            # 转换为float32类型
            signal_data = signal_data.astype(np.float32)
            
            # 移除NaN和无穷大值
            valid_mask = np.isfinite(signal_data)
            signal_data = signal_data[valid_mask]
            
            return signal_data
            
        except Exception as e:
            print(f"Error cleaning signal data: {e}")
            return np.array([], dtype=np.float32)

    def _load_all_data(self):
        """
        加载所有数据
        """
        all_data = []
        all_labels = []

        # HUST数据集的根路径
        hust_root_path = os.path.join(self.args.root_path, 'HUST')

        print(f"Loading HUST dataset from: {hust_root_path}")
        print(f"Loading workloads: {getattr(self.args, 'hust_workloads', ['all'])}")
        print(f"Using signal type: {getattr(self.args, 'hust_signal_type', 'x')}")

        # 获取工况列表
        workloads = getattr(self.args, 'hust_workloads', ['all'])
        if workloads is None:
            workloads = ['all']

        # 加载各工况数据
        for workload in workloads:
            # 获取文件信息
            files_info = self._get_workload_files(hust_root_path, workload)

            if not files_info:
                print(f"Warning: No valid files found for workload {workload}, skipping...")
                continue

            # 处理每个文件
            for file_path, fault_type, file_workload in files_info:
                try:
                    # 读取Excel文件
                    print(f"Processing {os.path.basename(file_path)}...")
                    df = self._read_excel_with_multiple_engines(file_path)
                    
                    print(f"Excel file shape: {df.shape}")
                    print(f"Column names: {list(df.columns[:10])}...")  # 打印前10列名
                    
                    # 根据您的描述，从第23行开始，第3,4,5列是xyz方向的加速度信号
                    start_row = 22  # 第23行（0索引）
                    
                    # 检查数据是否足够
                    if df.shape[0] <= start_row:
                        print(f"Warning: File {os.path.basename(file_path)} has insufficient rows")
                        continue
                        
                    if df.shape[1] < 5:
                        print(f"Warning: File {os.path.basename(file_path)} has insufficient columns")
                        continue
                    
                    x_col = 2  # 第3列（0索引）
                    y_col = 3  # 第4列（0索引）
                    z_col = 4  # 第5列（0索引）
                    
                    # 提取信号数据
                    signal_type = getattr(self.args, 'hust_signal_type', 'x')
                    
                    if signal_type == 'x':
                        raw_signal = df.iloc[start_row:, x_col].values
                        signal_data = self._clean_signal_data(raw_signal)
                    elif signal_type == 'y':
                        raw_signal = df.iloc[start_row:, y_col].values
                        signal_data = self._clean_signal_data(raw_signal)
                    elif signal_type == 'z':
                        raw_signal = df.iloc[start_row:, z_col].values
                        signal_data = self._clean_signal_data(raw_signal)
                    elif signal_type == 'xyz':
                        # 组合三个方向的信号
                        x_data = self._clean_signal_data(df.iloc[start_row:, x_col].values)
                        y_data = self._clean_signal_data(df.iloc[start_row:, y_col].values)
                        z_data = self._clean_signal_data(df.iloc[start_row:, z_col].values)
                        
                        # 确保三个方向的数据长度一致
                        min_len = min(len(x_data), len(y_data), len(z_data))
                        if min_len == 0:
                            print(f"Warning: No valid data in {os.path.basename(file_path)}")
                            continue
                            
                        x_data = x_data[:min_len]
                        y_data = y_data[:min_len]
                        z_data = z_data[:min_len]
                        
                        signal_data = np.sqrt(x_data**2 + y_data**2 + z_data**2)  # 向量模长
                    else:
                        raw_signal = df.iloc[start_row:, x_col].values  # 默认使用x方向
                        signal_data = self._clean_signal_data(raw_signal)

                    # 检查清理后的数据
                    if len(signal_data) == 0:
                        print(f"Warning: No valid signal data in {os.path.basename(file_path)} after cleaning")
                        continue
                    
                    if len(signal_data) < self.args.window_size:
                        print(f"Warning: Signal length {len(signal_data)} is smaller than window size {self.args.window_size}")
                        continue

                    print(f"Valid data length for {os.path.basename(file_path)}: {len(signal_data)}")

                    # 滑动窗口采样
                    windows = self._sliding_window(signal_data)

                    if len(windows) == 0:
                        print(f"Warning: No windows generated for {os.path.basename(file_path)}")
                        continue

                    # 根据任务类型设置标签
                    if fault_type not in self.fault_patterns:
                        print(f"Warning: Unknown fault type {fault_type}, skipping...")
                        continue
                        
                    fault_config = self.fault_patterns[fault_type]
                    task_type = getattr(self.args, 'hust_task_type', '9class')
                    
                    if task_type == '4class':
                        label = fault_config['class_4']
                        # 忽略复合故障（标签为-1）
                        if label == -1:
                            print(f"Skipping combination fault {fault_type} in 4-class mode")
                            continue
                    else:
                        label = fault_config['class_9']

                    # 添加数据
                    all_data.extend(windows)
                    all_labels.extend([label] * len(windows))

                    print(f"Processed {os.path.basename(file_path)}: {len(windows)} windows, "
                          f"label: {label} ({fault_config['name']})")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 转换为numpy数组
        if len(all_data) == 0:
            raise ValueError(f"No data loaded for HUST dataset. "
                           f"Please check your data path and file format.")

        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        return all_data, all_labels

    def _load_data(self):
        """加载数据"""
        # 加载所有数据
        all_data, all_labels = self._load_all_data()

        # 打乱数据（使用固定随机种子确保一致性）
        np.random.seed(42)
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]
        all_labels = all_labels[indices]

        # 划分数据集
        n_samples = len(all_data)
        train_size = int(n_samples * self.args.train_ratio)
        val_size = int(n_samples * self.args.val_ratio)

        # 获取训练集数据（用于scaler初始化）
        train_data = all_data[:train_size]
        train_labels = all_labels[:train_size]

        # 如果需要标准化，使用训练集数据初始化scaler
        if self.args.normalize and self.scaler is not None:
            print("Initializing scaler with training data...")
            train_data_reshaped = train_data.reshape(-1, self.args.window_size)
            self.scaler.fit(train_data_reshaped)
            print("Scaler fitted successfully.")

        # 根据flag选择对应的数据子集
        if self.flag == 'train':
            self.data = train_data
            self.labels = train_labels
        elif self.flag == 'val':
            self.data = all_data[train_size:train_size + val_size]
            self.labels = all_labels[train_size:train_size + val_size]
        else:  # test
            self.data = all_data[train_size + val_size:]
            self.labels = all_labels[train_size + val_size:]

        task_type = getattr(self.args, 'hust_task_type', '9class')
        print(f"Loaded {self.flag} dataset: {len(self.data)} samples, "
              f"{task_type} ({self.num_classes} classes)")

        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        class_dist = {}
        for u, c in zip(unique, counts):
            if u < len(self.class_names):
                class_dist[f"{self.class_names[u]}({u})"] = c
        print(f"Class distribution: {class_dist}")

    def _sliding_window(self, signal_data):
        """滑动窗口采样"""
        windows = []
        for i in range(0, len(signal_data) - self.args.window_size + 1, self.args.stride):
            window = signal_data[i:i + self.args.window_size]
            windows.append(window)
        return windows

    def __getitem__(self, idx):
        """获取单个样本"""
        signal_window = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(signal_window), torch.LongTensor([label]).squeeze()

    def __len__(self):
        """数据集长度"""
        return len(self.data)

    def get_class_names(self):
        """获取类别名称"""
        return self.class_names