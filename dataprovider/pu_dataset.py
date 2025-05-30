import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class PUBearingDataset(Dataset):
    """
    PU Bearing Fault Diagnosis Dataset
    支持多种分类任务和多种信号类型
    """

    def __init__(self, args, flag='train'):
        """
        Args:
            args: 命令行参数对象
            flag: 'train', 'val', 'test'
            scaler: 预训练的StandardScaler（用于val/test集）
        """
        self.args = args
        self.args.sampling_rate = 64000  # PU数据集的采样率固定为64000Hz
        self.flag = flag


        # 初始化标准化器
        if args.normalize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # 定义PU数据集的类别映射
        self._setup_pu_classes()

        # 加载数据
        self._load_data()
        # 如果需要标准化，则对所有数据进行标准化
        if self.args.normalize and self.scaler is not None:
            self.data = self._standardize_data(self.data)

        # 如果有变换类型，则提前应用变换
        if self.args.transform_type != 'None':
            self.data = self._apply_transform_to_all_data(self.data)

    def _standardize_data(self, data):
        standardized_data = []
        for signal_window in data:
            if len(signal_window.shape) == 1:  # 单信号
                standardized_signal = self.scaler.transform(signal_window.reshape(1, -1)).reshape(-1)
            else:  # 多信号 (n_channels, window_size)
                original_shape = signal_window.shape
                standardized_signal = self.scaler.transform(signal_window.reshape(1, -1)).reshape(original_shape)
            standardized_data.append(standardized_signal)
        return np.array(standardized_data)

    def _apply_transform_to_all_data(self, data):
        transformed_data = []
        for signal_window in data:
            if len(signal_window.shape) == 1:  # 单信号
                transformed_signal = apply_transform(signal_window, self.args.sampling_rate, self.args)
                # 确保输出是2D格式并添加通道维度
                if len(transformed_signal.shape) == 2:
                    transformed_signal = np.expand_dims(transformed_signal, axis=0)
            else:  # 多信号，对每个信号分别变换
                transformed_signals = []
                for i in range(signal_window.shape[0]):
                    transformed = apply_transform(signal_window[i], self.args.sampling_rate, self.args)
                    if len(transformed.shape) == 2:
                        transformed_signals.append(transformed)
                    else:
                        transformed_signals.append(np.expand_dims(transformed, axis=0))

                transformed_signal = np.stack(transformed_signals, axis=0)
            transformed_data.append(transformed_signal)
        return np.array(transformed_data)

    def _setup_pu_classes(self):
        """设置PU数据集的类别映射"""
        # PU人工故障分类定义
        if self.args.pu_task_type == '3class_artificial':
            # 简单三分类：正常、内圈故障、外圈故障
            self.class_mapping = {
                'K001': 0,  # 正常
                'KI01': 1, 'KI03': 1, 'KI05': 1, 'KI07': 1, 'KI08': 1,  # 内圈故障
                'KA01': 2, 'KA03': 2, 'KA05': 2, 'KA07': 2, 'KA09': 2,  # 外圈故障
                 # 内圈故障
            }
            self.class_names = ['normal', 'outer_ring', 'inner_ring']
            self.num_classes = 3
            # 根据任务类型确定需要的轴承类型
            self.required_bearing_types = ['K001', 'KA01', 'KA03', 'KA05', 'KA07', 'KA09',
                                           'KI01', 'KI03', 'KI05', 'KI07', 'KI08']

        elif self.args.pu_task_type == '5class_artificial':
            # 五分类
            self.class_mapping = {
                'K001': 0,  # 正常
                'KA05': 1, 'KA07': 1,  # 外圈故障类别1
                'KA03': 2, 'KA06': 2, 'KA08': 2, 'KA09': 2,  # 外圈故障类别2
                'KI01': 3, 'KI03': 3, 'KI05': 3,  # 内圈故障类别1
                'KI07': 4, 'KI08': 4  # 内圈故障类别2
            }
            self.class_names = ['normal', 'outer_ring_1', 'outer_ring_2', 'inner_ring_1', 'inner_ring_2']
            self.num_classes = 5
            self.required_bearing_types = ['K001', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09',
                                           'KI01', 'KI03', 'KI05', 'KI07', 'KI08']

        elif self.args.pu_task_type == '9class_artificial':
            # 九分类
            self.class_mapping = {
                'K001': 0,  # 正常
                'KA01': 1, 'KA03': 2, 'KA07': 3, 'KA05': 4, 'KA09': 5,  # 外圈故障详细分类
                'KI01': 6, 'KI03': 7, 'KI08': 8  # 内圈故障详细分类
            }
            self.class_names = ['normal', 'KA01', 'KA03', 'KA07', 'KA05', 'KA09',
                                'KI01', 'KI03', 'KI08']
            self.num_classes = 9
            self.required_bearing_types = ['K001', 'KA01', 'KA03', 'KA05', 'KA07', 'KA09',
                                           'KI01', 'KI03', 'KI08']

        elif self.args.pu_task_type == '13class_artificial':
            # 十三分类
            self.class_mapping = {
                'K001': 0,  # 正常
                'KA01': 1, 'KA03': 2, 'KA05': 3, 'KA06': 4, 'KA07': 5,
                'KA08': 6, 'KA09': 7,  # 外圈故障详细分类
                'KI01': 8, 'KI03': 9, 'KI05': 10, 'KI07': 11, 'KI08': 12  # 内圈故障详细分类
            }
            self.class_names = ['normal', 'KA01', 'KA03', 'KA05', 'KA06', 'KA07',
                                'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08']
            self.num_classes = 13
            self.required_bearing_types = ['K001', 'KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09',
                                           'KI01', 'KI03', 'KI05', 'KI07', 'KI08']

        elif self.args.pu_task_type == '15class_nature':
            # 自然故障十五分类
            self.class_mapping = {
                'K002': 0,  # 正常
                'KA04': 1, 'KA15': 2, 'KA22': 3, 'KA30': 4, 'KA16': 5,  # 外圈自然故障
                'KB23': 6, 'KB24': 7, 'KB27': 8,  # 混合故障
                'KI04': 9, 'KI14': 10, 'KI16': 11, 'KI17': 12, 'KI18': 13, 'KI21': 14  # 内圈自然故障
            }
            self.class_names = ['normal', 'KA04', 'KA15', 'KA22', 'KA30', 'KA16',
                                'KB23', 'KB24', 'KB27', 'KI04', 'KI14', 'KI16',
                                'KI17', 'KI18', 'KI21']
            self.num_classes = 15
            self.required_bearing_types = ['K002', 'KA04', 'KA15', 'KA22', 'KA30', 'KA16',
                                           'KB23', 'KB24', 'KB27', 'KI04', 'KI14', 'KI16',
                                           'KI17', 'KI18', 'KI21']

    def _load_data(self):
        """加载PU数据"""
        all_data = []
        all_labels = []

        print(f"Loading PU dataset with task type: {self.args.pu_task_type}")
        print(f"Signal type: {self.args.pu_signal_type}")
        print(f"Required bearing types: {self.required_bearing_types}")
        print(f"Workloads: {self.args.pu_workloads}")

        # PU数据集的根路径
        pu_root_path = os.path.join(self.args.root_path, 'PU')

        # 遍历任务所需的轴承类型
        for bearing_type in self.required_bearing_types:
            bearing_path = os.path.join(pu_root_path, bearing_type)

            if not os.path.exists(bearing_path):
                print(f"Warning: Bearing path '{bearing_path}' does not exist, skipping...")
                continue

            # 检查该轴承类型是否在当前任务的类别映射中
            if bearing_type not in self.class_mapping:
                print(f"Warning: Bearing type '{bearing_type}' not in task '{self.args.pu_task_type}', skipping...")
                continue

            label = self.class_mapping[bearing_type]

            # 获取所有mat文件
            mat_files = [f for f in os.listdir(bearing_path) if f.endswith('.mat')]
            mat_files.sort()

            print(f"Processing bearing type: {bearing_type}, found {len(mat_files)} files")

            for mat_file in mat_files:
                # 检查文件名是否包含指定的工况
                file_workload = self._extract_workload_from_filename(mat_file)
                if file_workload not in self.args.pu_workloads:
                    continue

                file_path = os.path.join(bearing_path, mat_file)

                try:
                    # 读取mat文件
                    mat_data = loadmat(file_path)

                    # 获取文件名（不含扩展名）作为数据键
                    file_key = os.path.splitext(mat_file)[0]

                    if file_key not in mat_data:
                        print(f"Warning: Key '{file_key}' not found in {mat_file}, skipping...")
                        continue

                    # 根据你提供的索引方式提取信号数据
                    signal_data = self._extract_pu_signal_data(mat_data[file_key], mat_file)
                    if signal_data is None:
                        continue

                    print(f"Processing {mat_file}: signal shape = {signal_data.shape}, label = {label}")

                    # 滑动窗口采样
                    windows = self._sliding_window(signal_data)

                    # 添加数据
                    all_data.extend(windows)
                    all_labels.extend([label] * len(windows))

                    print(f"Generated {len(windows)} windows from {mat_file}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        # 转换为numpy数组
        if len(all_data) == 0:
            raise ValueError(f"No data loaded for PU dataset with task type '{self.args.pu_task_type}' "
                             f"and bearing types '{self.args.pu_bearing_types}'. Please check your data path.")

        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        # 打乱数据
        np.random.seed(42)
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]
        all_labels = all_labels[indices]

        # 划分数据集
        n_samples = len(all_data)
        train_size = int(n_samples * self.args.train_ratio)
        val_size = int(n_samples * self.args.val_ratio)

        if self.flag == 'train':
            self.data = all_data[:train_size]
            self.labels = all_labels[:train_size]
        elif self.flag == 'val':
            self.data = all_data[train_size:train_size + val_size]
            self.labels = all_labels[train_size:train_size + val_size]
        else:  # test
            self.data = all_data[train_size + val_size:]
            self.labels = all_labels[train_size + val_size:]

        # 获取训练集数据（用于scaler初始化）
        train_data = all_data[:train_size]

        # 如果需要标准化，使用训练集数据初始化scaler
        if self.args.normalize and self.scaler is not None:
            print("Initializing scaler with training data...")
            train_data_reshaped = train_data.reshape(-1, self.args.window_size)
            self.scaler.fit(train_data_reshaped)
            print("Scaler fitted successfully.")

        print(f"Loaded PU {self.flag} dataset: {len(self.data)} samples, "
              f"{self.args.pu_task_type} ({self.num_classes} classes)")

        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

    def _extract_workload_from_filename(self, filename):
        """从文件名提取工况信息"""
        # 文件名格式: N09_M07_F10_KA01_1.mat
        # 提取前三部分: N09_M07_F10
        parts = filename.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:3])
        return None

    def _extract_pu_signal_data(self, mat_data, filename):
        try:
            # 振动信号: mdata[0][0][2][0][6][2]
            # 电流信号: mdata[0][0][2][0][1][2], mdata[0][0][2][0][2][2]

            vibration_data = None
            current_data = None

            # 提取振动信号
            try:
                vibration_data = mat_data[0][0][2][0][6][2].reshape(-1)
            except (IndexError, AttributeError):
                print(f"Warning: Cannot extract vibration signal from {filename}")

            # 提取电流信号
            try:
                current_1 = mat_data[0][0][2][0][1][2].reshape(-1)
                current_2 = mat_data[0][0][2][0][2][2].reshape(-1)
                # 合并两列电流信号
                if len(current_1) == len(current_2):
                    current_data = np.stack([current_1, current_2], axis=0)
                else:
                    # 如果长度不同，取较短的长度
                    min_len = min(len(current_1), len(current_2))
                    current_data = np.stack([current_1[:min_len], current_2[:min_len]], axis=0)
            except (IndexError, AttributeError):
                print(f"Warning: Cannot extract current signal from {filename}")

            # 根据选择的信号类型返回数据
            if self.args.pu_signal_type == 'vibration':
                if vibration_data is not None:
                    return vibration_data
                else:
                    print(f"Warning: No vibration signal found in {filename}")
                    return None
            elif self.args.pu_signal_type == 'current':
                if current_data is not None:
                    return current_data
                else:
                    print(f"Warning: No current signal found in {filename}")
                    return None
            elif self.args.pu_signal_type == 'both':
                if vibration_data is not None and current_data is not None:
                    # 确保振动信号和电流信号长度相同
                    min_len = min(len(vibration_data), current_data.shape[1])
                    # 合并振动和电流信号 (3, length)
                    return np.concatenate([
                        vibration_data[:min_len].reshape(1, -1),
                        current_data[:, :min_len]
                    ], axis=0)
                else:
                    print(f"Warning: Both signals not available in {filename}")
                    return None

        except Exception as e:
            print(f"Error extracting signal data from {filename}: {e}")
            return None

        return None

    def _sliding_window(self, signal_data):
        """滑动窗口采样"""
        windows = []

        if len(signal_data.shape) == 1:  # 单信号
            for i in range(0, len(signal_data) - self.args.window_size + 1, self.args.stride):
                window = signal_data[i:i + self.args.window_size]
                windows.append(window)
        else:  # 多信号 (n_channels, length)
            signal_length = signal_data.shape[1]
            for i in range(0, signal_length - self.args.window_size + 1, self.args.stride):
                window = signal_data[:, i:i + self.args.window_size]
                windows.append(window)

        return windows

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
