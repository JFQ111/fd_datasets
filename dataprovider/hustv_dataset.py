import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform
import os.path as op


class HUSTVBearingDataset(Dataset):
    """
    HUST(V) Bearing Fault Diagnosis Dataset
    支持4分类和7分类任务，支持多工况组合，支持多轴承选择
    基于51200Hz采样率
    """

    def __init__(self, args, flag='train'):
        """
        Args:
            args: 命令行参数对象
            flag: 'train', 'val', 'test'
        """
        self.args = args
        self.flag = flag
        # HUST(V)数据集固定采样率
        self.args.sampling_rate = 51200
        
        # 初始化标准化器
        if args.normalize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # 定义轴承型号
        self.bearing_types = ['6204', '6205', '6206', '6207', '6208']
        
        # 定义工况
        self.workload_mapping = {'0': '0', '2': '2', '4': '4'}  # 对应0W, 200W, 400W
        
        # 定义故障类型映射 - 基于文件名前缀识别
        self.fault_patterns = {
            'N': {'type': 'normal', 'class_7': 0, 'class_4': 0},      # 正常
            'I': {'type': 'inner', 'class_7': 1, 'class_4': 1},       # 内圈
            'O': {'type': 'outer', 'class_7': 2, 'class_4': 2},       # 外圈
            'B': {'type': 'ball', 'class_7': 3, 'class_4': 3},        # 滚动体
            'IO': {'type': 'inner_outer', 'class_7': 4, 'class_4': -1},  # 内外圈复合 (4分类中忽略)
            'IB': {'type': 'inner_ball', 'class_7': 5, 'class_4': -1},   # 内圈滚动体复合 (4分类中忽略)
            'OB': {'type': 'outer_ball', 'class_7': 6, 'class_4': -1},   # 外圈滚动体复合 (4分类中忽略)
        }

        # 定义类别名称
        self.class_names_7 = ['normal', 'inner', 'outer', 'ball', 
                              'inner_outer', 'inner_ball', 'outer_ball']
        self.class_names_4 = ['normal', 'inner', 'outer', 'ball']

        # 设置当前任务的类别名称
        if args.hustv_task_type == '4class':
            self.class_names = self.class_names_4
            self.num_classes = 4
        else:
            self.class_names = self.class_names_7
            self.num_classes = 7

        # 加载数据
        self._load_data()
        
        # 如果需要标准化，则对所有数据进行标准化
        if self.args.normalize and self.scaler is not None:
            self.data = self.scaler.transform(self.data)

        # 如果有变换类型，则提前应用变换
        if self.args.transform_type != 'None':
            self.data = self._apply_transform_to_all_data()

    def _apply_transform_to_all_data(self):
        """应用变换到所有数据"""
        transformed_data = []
        for signal_window in self.data:
            transformed_signal = apply_transform(signal_window, self.args.sampling_rate, self.args)
            # 确保输出是2D格式并添加通道维度
            if len(transformed_signal.shape) == 2:
                transformed_signal = np.expand_dims(transformed_signal, axis=0)
            transformed_data.append(transformed_signal)
        return np.array(transformed_data)

    def _identify_fault_type(self, filename):
        """
        根据文件名识别故障类型
        
        Args:
            filename: 文件名
            
        Returns:
            fault_prefix: 故障类型前缀，如'N', 'I', 'O', 'B', 'IO', 'IB', 'OB'
        """
        # 提取文件名前缀（不包含扩展名）
        base_name = os.path.splitext(filename)[0]
        
        # 按长度降序排列，避免短模式匹配长模式的问题
        patterns = ['IO', 'IB', 'OB', 'I', 'O', 'B', 'N']
        
        for pattern in patterns:
            if base_name.startswith(pattern):
                return pattern
                
        return None

    def _get_bearing_files(self, bearing_type, workload):
        """
        获取指定轴承型号和工况下的所有mat文件
        
        Args:
            bearing_type: 轴承型号 (如 '6204')
            workload: 工况 (如 '0', '2', '4')
            
        Returns:
            files_info: [(file_path, fault_type), ...]
        """
        bearing_path = os.path.join(self.args.root_path, 'HUST_V', bearing_type, workload)
        files_info = []
        
        if not os.path.exists(bearing_path):
            print(f"Warning: Bearing path '{bearing_path}' does not exist.")
            return files_info
            
        # 获取所有mat文件
        mat_files = [f for f in os.listdir(bearing_path) if f.endswith('.mat')]
        mat_files.sort()  # 保证顺序一致性
        
        print(f"Found {len(mat_files)} mat files in {bearing_type}/{workload}")
        
        for mat_file in mat_files:
            file_path = os.path.join(bearing_path, mat_file)
            
            # 识别故障类型
            fault_prefix = self._identify_fault_type(mat_file)
            if fault_prefix is None:
                print(f"Warning: Unknown fault pattern in file '{mat_file}', skipping...")
                continue
                
            # 检查6204轴承的特殊情况 - 缺少B和IB故障
            if bearing_type == '6204' and fault_prefix in ['B', 'IB']:
                print(f"Info: Bearing {bearing_type} missing {fault_prefix} fault, skipping {mat_file}")
                continue
                
            files_info.append((file_path, fault_prefix))
            print(f"Loaded: {mat_file} -> {fault_prefix}")
            
        return files_info

    def _load_all_data(self):
        """
        加载所有数据（用于数据集划分和scaler初始化）
        
        Returns:
            all_data: 所有数据的numpy array
            all_labels: 所有标签的numpy array
        """
        all_data = []
        all_labels = []
        
        print(f"Loading HUST(V) dataset...")
        print(f"Bearings: {self.args.hustv_bearings}")
        print(f"Workloads: {self.args.hustv_workloads}")
        print(f"Task type: {self.args.hustv_task_type}")
        print(f"Sampling rate: {self.args.sampling_rate}Hz")
        
        # 加载各轴承和工况数据
        for bearing_type in self.args.hustv_bearings:
            if bearing_type not in self.bearing_types:
                print(f"Warning: Unknown bearing type '{bearing_type}', skipping...")
                continue
                
            for workload in self.args.hustv_workloads:
                if workload not in self.workload_mapping:
                    print(f"Warning: Unknown workload '{workload}', skipping...")
                    continue
                    
                # 获取文件信息
                files_info = self._get_bearing_files(bearing_type, workload)
                
                if not files_info:
                    print(f"Warning: No valid files found in {bearing_type}/{workload}, skipping...")
                    continue
                    
                # 处理每个文件
                for file_path, fault_prefix in files_info:
                    try:
                        # 读取mat文件 - 使用您提供的读取逻辑
                        data_dict = loadmat(file_path)
                        signal_data = data_dict['data'].reshape(-1)
                        
                        print(f"Original data length for {os.path.basename(file_path)}: {len(signal_data)}")
                        
                        # 数据预处理 - 每4个点取1个（降采样）
                        signal_data = signal_data[::4]
                        
                        # 根据故障类型设置数据长度
                        if fault_prefix == 'N':  # 正常数据使用更多样本
                            para_length = self.args.window_size
                            num_samples = 500  # 正常数据更多样本
                            max_length = int(len(signal_data) * 0.8)  # 使用80%的数据
                        else:  # 故障数据
                            para_length = self.args.window_size
                            num_samples = 300  # 故障数据较少样本
                            max_length = int(len(signal_data) * 0.6)  # 使用60%的数据
                            
                        # 确保有足够的数据
                        if max_length < para_length:
                            print(f"Warning: Not enough data in {os.path.basename(file_path)}, skipping...")
                            continue
                            
                        # 生成采样位置
                        sample_positions = np.linspace(0, max_length - para_length, num_samples, dtype=int)
                        
                        # 提取窗口数据
                        windows = []
                        for pos in sample_positions:
                            window = signal_data[pos:pos + para_length]
                            windows.append(window)
                            
                        # 添加数据
                        all_data.extend(windows)
                        
                        # 根据任务类型设置标签
                        fault_config = self.fault_patterns[fault_prefix]
                        if self.args.hustv_task_type == '4class':
                            label = fault_config['class_4']
                            # 跳过复合故障（标签为-1）
                            if label == -1:
                                print(f"Skipping compound fault {fault_prefix} for 4-class task")
                                # 移除刚添加的数据
                                all_data = all_data[:-len(windows)]
                                continue
                        else:
                            label = fault_config['class_7']
                            
                        all_labels.extend([label] * len(windows))
                        
                        print(f"Processed {os.path.basename(file_path)}: {len(windows)} windows, "
                              f"label: {label} ({fault_config['type']})")
                              
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
                        
        # 转换为numpy数组
        if len(all_data) == 0:
            raise ValueError(f"No data loaded for bearings '{self.args.hustv_bearings}' "
                           f"and workloads '{self.args.hustv_workloads}'. Please check your data path.")
                           
        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        return all_data, all_labels

    def _load_data(self):
        """加载数据"""
        # 加载所有数据
        all_data, all_labels = self._load_all_data()
        
        # 打乱数据（使用固定随机种子确保一致性）
        np.random.seed(42)  # 固定种子确保train/val/test划分一致
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
            
        print(f"Loaded {self.flag} dataset: {len(self.data)} samples, "
              f"{self.args.hustv_task_type} ({self.num_classes} classes)")
              
        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Class distribution: {class_distribution}")
        
        # 打印类别对应关系
        print("Class mapping:")
        for i, class_name in enumerate(self.class_names):
            if i in class_distribution:
                print(f"  {i}: {class_name} ({class_distribution[i]} samples)")

    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取预处理后的信号
        signal_window = self.data[idx]
        label = self.labels[idx]
        # 返回数据和标签
        return torch.FloatTensor(signal_window), torch.LongTensor([label]).squeeze()

    def __len__(self):
        """数据集长度"""
        return len(self.data)

    def get_class_names(self):
        """获取类别名称"""
        return self.class_names