import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
from transforms.signal_transforms import apply_transform


class MFPTBearingDataset(Dataset):
    """
    MFPT Bearing Fault Diagnosis Dataset
    支持3分类任务：正常(0)、内圈故障(1)、外圈故障(2)
    """

    def __init__(self, args, flag='train'):
        """
        Args:
            args: 命令行参数对象
            flag: 'train', 'val', 'test'
        """
        self.args = args
        self.flag = flag

        # 初始化标准化器
        if args.normalize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # 定义文件夹和对应的故障类型
        self.folder_mapping = {
            '1 - Three Baseline Conditions': {
                'fault_type': 'normal',
                'class_label': 0,
                'file_prefix': 'baseline_',
                'sampling_rate': 97656,
                'description': '正常状态'
            },
            '2 - Three Outer Race Fault Conditions': {
                'fault_type': 'outer_race',
                'class_label': 2,
                'file_prefix': 'OuterRaceFault_',
                'sampling_rate': 97656,
                'description': '外圈故障'
            },
            '3 - Seven More Outer Race Fault Conditions': {
                'fault_type': 'outer_race',
                'class_label': 2,
                'file_prefix': 'OuterRaceFault_vload_',
                'sampling_rate': 48828,
                'description': '外圈故障(不同负载)'
            },
            '4 - Seven Inner Race Fault Conditions': {
                'fault_type': 'inner_race',
                'class_label': 1,
                'file_prefix': 'InnerRaceFault_vload_',
                'sampling_rate': 48828,
                'description': '内圈故障(不同负载)'
            }
        }

        # 定义目标采样率（统一重采样到48828Hz）
        self.target_sampling_rate = 48828
        # 定义类别名称
        self.class_names = ['normal', 'inner_race', 'outer_race']

        self.num_classes = 3
        self._load_data()

        print(f"Target sampling rate: {self.target_sampling_rate} Hz")
        # 如果需要标准化，则对所有数据进行标准化
        if self.args.normalize and self.scaler is not None:
            self.data = self.scaler.transform(self.data)

        # 如果有变换类型，则提前应用变换
        if self.args.transform_type != 'None':
            self.data = self._apply_transform_to_all_data()

    def _apply_transform_to_all_data(self):
        transformed_data = []
        for signal_window in self.data:
            transformed_signal = apply_transform(signal_window, self.args.sampling_rate, self.args)
            # 确保输出是2D格式并添加通道维度
            if len(transformed_signal.shape) == 2:
                transformed_signal = np.expand_dims(transformed_signal, axis=0)
            transformed_data.append(transformed_signal)
        return np.array(transformed_data)

    def _get_folder_files(self, mfpt_root_path, folder_name):
        """
        获取指定文件夹下的所有mat文件信息

        Args:
            mfpt_root_path: MFPT数据集根路径
            folder_name: 文件夹名称

        Returns:
            files_info: [(file_path, fault_type, class_label, sampling_rate), ...]
        """
        folder_path = os.path.join(mfpt_root_path, folder_name)
        files_info = []

        if not os.path.exists(folder_path):
            print(f"Warning: Folder path '{folder_path}' does not exist.")
            return files_info

        folder_config = self.folder_mapping[folder_name]

        # 获取所有mat文件
        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        mat_files.sort()  # 保证顺序一致性

        print(f"Found {len(mat_files)} mat files in {folder_name}")
        print(f"  - Fault type: {folder_config['description']}")
        print(f"  - Sampling rate: {folder_config['sampling_rate']} Hz")

        for mat_file in mat_files:
            file_path = os.path.join(folder_path, mat_file)
            files_info.append((
                file_path,
                folder_config['fault_type'],
                folder_config['class_label'],
                folder_config['sampling_rate']
            ))
            print(f"  - {mat_file}")

        return files_info

    def _extract_bearing_data(self, mat_data, filename):
        """
        从MFPT mat文件中提取轴承振动数据
        MFPT数据结构：data['bearing'][0][0][1] 包含振动数据

        Args:
            mat_data: loadmat返回的数据字典
            filename: 文件名（用于调试）

        Returns:
            signal_data: 提取的振动信号数据，如果提取失败则返回None
        """
        try:
            # MFPT数据集的标准结构：data['bearing'][0][0][1]
            if 'bearing' in mat_data:
                bearing_data = mat_data['bearing']
                # 根据您提供的结构访问振动数据
                for idx in [0, 1, 2]:
                    signal_data = bearing_data[0][0][idx]
                    if signal_data.shape[0]>1024:
                        break

                # 处理数据形状，转换为1D数组
                if len(signal_data.shape) == 2 and signal_data.shape[1] == 1:
                    signal_data = signal_data.reshape(-1)
                elif len(signal_data.shape) == 2:
                    # 如果有多列，取第一列
                    signal_data = signal_data[:, 0]
                else:
                    signal_data = signal_data.reshape(-1)

                print(f"Successfully extracted bearing data from '{filename}': {signal_data.shape}")
                return signal_data
            else:
                print(f"Warning: 'bearing' key not found in '{filename}'")
                return None

        except Exception as e:
            print(f"Error extracting bearing data from '{filename}': {e}")
            # 尝试其他可能的数据结构
            try:
                # 查找其他可能的数据键
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and data.size > 10000:
                            if len(data.shape) == 2 and data.shape[1] == 1:
                                return data.reshape(-1)
                            elif len(data.shape) == 1:
                                return data
                print(f"Could not find suitable data in '{filename}'")
                return None
            except:
                print(f"Failed to extract any data from '{filename}'")
                return None

    def _resample_signal(self, signal_data, original_sampling_rate):
        """
        将信号重采样到目标采样率

        Args:
            signal_data: 原始信号数据
            original_sampling_rate: 原始采样率

        Returns:
            resampled_signal: 重采样后的信号数据
        """
        if original_sampling_rate == self.target_sampling_rate:
            return signal_data

        # 计算重采样比例
        resample_ratio = self.target_sampling_rate / original_sampling_rate
        new_length = int(len(signal_data) * resample_ratio)

        # 使用scipy.signal.resample进行重采样
        resampled_signal = resample(signal_data, new_length)

        print(f"Resampled from {original_sampling_rate}Hz to {self.target_sampling_rate}Hz: "
              f"{len(signal_data)} -> {len(resampled_signal)} samples")

        return resampled_signal

    def _load_all_data(self):
        """
        加载所有数据（用于数据集划分和scaler初始化）

        Returns:
            all_data: 所有数据的numpy array
            all_labels: 所有标签的numpy array
        """
        all_data = []
        all_labels = []

        # MFPT数据集的根路径
        mfpt_root_path = os.path.join(self.args.root_path, 'MFPT')

        if not os.path.exists(mfpt_root_path):
            raise ValueError(f"MFPT root path '{mfpt_root_path}' does not exist.")

        print(f"Loading MFPT dataset from: {mfpt_root_path}")
        print(f"Target sampling rate: {self.target_sampling_rate} Hz")
        print(f"Window size: {self.args.window_size}, Stride: {self.args.stride}")

        # 遍历所有文件夹
        for folder_name in self.folder_mapping.keys():
            files_info = self._get_folder_files(mfpt_root_path, folder_name)

            if not files_info:
                print(f"Warning: No valid files found in {folder_name}, skipping...")
                continue

            # 处理每个文件
            for file_path, fault_type, class_label, sampling_rate in files_info:
                try:
                    # 读取mat文件
                    mat_data = loadmat(file_path)

                    # 使用新的数据提取方法
                    signal_data = self._extract_bearing_data(mat_data, os.path.basename(file_path))

                    if signal_data is None:
                        print(f"Warning: Failed to extract data from '{os.path.basename(file_path)}', skipping...")
                        continue

                    print(f"Original data length for {os.path.basename(file_path)}: {len(signal_data)} "
                          f"(sampling rate: {sampling_rate}Hz)")

                    # 重采样到目标采样率
                    signal_data = self._resample_signal(signal_data, sampling_rate)

                    # 滑动窗口采样
                    windows = self._sliding_window(signal_data)

                    if len(windows) == 0:
                        print(f"Warning: No windows generated from '{os.path.basename(file_path)}', skipping...")
                        continue

                    # 添加数据
                    all_data.extend(windows)
                    all_labels.extend([class_label] * len(windows))

                    print(f"Processed {os.path.basename(file_path)}: {len(windows)} windows, "
                          f"label: {class_label} ({fault_type})")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        # 转换为numpy数组
        if len(all_data) == 0:
            raise ValueError("No data loaded. Please check your data path and file formats.")

        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        print(f"Total loaded: {len(all_data)} samples")
        print(f"Data shape: {all_data.shape}")

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
              f"3-class classification ({self.num_classes} classes)")

        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        class_dist = {self.class_names[i]: counts[np.where(unique == i)[0][0]] if i in unique else 0
                      for i in range(self.num_classes)}
        print(f"Class distribution: {class_dist}")

    def _sliding_window(self, signal_data):
        """滑动窗口采样"""
        windows = []
        window_size = self.args.window_size
        stride = self.args.stride

        if len(signal_data) < window_size:
            print(f"Warning: Signal length {len(signal_data)} is shorter than window size {window_size}")
            return windows

        for i in range(0, len(signal_data) - window_size + 1, stride):
            window = signal_data[i:i + window_size]
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

    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'dataset_name': 'MFPT',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_mapping': {
                0: 'normal',
                1: 'inner_race',
                2: 'outer_race'
            },
            'total_samples': len(self.data),
            'sampling_rate': self.target_sampling_rate,
            'window_size': self.args.window_size,
            'stride': self.args.stride
        }
        return info