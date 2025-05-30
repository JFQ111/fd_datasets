import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class CWRUBearingDataset(Dataset):
    """
    CWRU Bearing Fault Diagnosis Dataset
    支持4分类和10分类任务，支持多工况组合，支持12k和48k采样率
    改进版：无论flag是什么，都自动使用训练集数据初始化scaler
    """

    def __init__(self, args, flag='train'):
        """
        Args:
            args: 命令行参数对象
            flag: 'train', 'val', 'test'
        """
        self.args = args
        #采样率
        if args.data_source == '12k_DE':
            self.args.sampling_rate= 12000
        elif args.data_source == '48k_DE':
            self.args.sampling_rate = 48000
        self.flag = flag

        # 初始化标准化器
        if args.normalize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # 定义类别映射 - 基于文件名模式识别
        self.fault_patterns = {
            'normal': {'pattern': 'normal', 'class_10': 0, 'class_4': 0},
            'IR007': {'pattern': 'IR007', 'class_10': 1, 'class_4': 1},  # 内圈0.007
            'B007': {'pattern': 'B007', 'class_10': 2, 'class_4': 3},  # 滚动体0.007
            'OR007': {'pattern': 'OR007', 'class_10': 3, 'class_4': 2},  # 外圈0.007
            'IR014': {'pattern': 'IR014', 'class_10': 4, 'class_4': 1},  # 内圈0.014
            'B014': {'pattern': 'B014', 'class_10': 5, 'class_4': 3},  # 滚动体0.014
            'OR014': {'pattern': 'OR014', 'class_10': 6, 'class_4': 2},  # 外圈0.014
            'IR021': {'pattern': 'IR021', 'class_10': 7, 'class_4': 1},  # 内圈0.021
            'B021': {'pattern': 'B021', 'class_10': 8, 'class_4': 3},  # 滚动体0.021
            'OR021': {'pattern': 'OR021', 'class_10': 9, 'class_4': 2},  # 外圈0.021
        }

        # 定义类别名称
        self.class_names_10 = ['normal', 'inner_7', 'ball_7', 'outer_7',
                               'inner_14', 'ball_14', 'outer_14',
                               'inner_21', 'ball_21', 'outer_21']

        self.class_names_4 = ['normal', 'inner', 'ball', 'outer']

        # 设置当前任务的类别名称
        if args.task_type == '4class':
            self.class_names = self.class_names_4
            self.num_classes = 4
        else:
            self.class_names = self.class_names_10
            self.num_classes = 10

        # 加载数据
        self._load_data()
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

    def _get_workload_files(self, data_source_path, workload):
        """
        自动获取指定数据源和工况下的所有mat文件及其对应的DE数据列

        Args:
            data_source_path: 数据源路径 (如 ./datasets/CWRU/12k_DE)
            workload: 工况名称 (如 '0hp')

        Returns:
            files_info: [(file_path, data_column, fault_type), ...]
        """
        workload_path = os.path.join(data_source_path, workload)
        files_info = []

        if not os.path.exists(workload_path):
            print(f"Warning: Workload path '{workload_path}' does not exist.")
            return files_info

        # 获取所有mat文件
        mat_files = [f for f in os.listdir(workload_path) if f.endswith('.mat')]
        mat_files.sort()  # 保证顺序一致性

        print(f"Found {len(mat_files)} mat files in {workload_path} (sampling rate: {self.args.sampling_rate}Hz)")

        for mat_file in mat_files:
            file_path = os.path.join(workload_path, mat_file)

            # 识别故障类型
            fault_type = self._identify_fault_type(mat_file)
            if fault_type is None:
                print(f"Warning: Unknown fault pattern in file '{mat_file}', skipping...")
                continue

            # 读取mat文件并查找DE数据列
            try:
                mat_data = loadmat(file_path)
                de_column = self._find_de_column(mat_data, mat_file)

                if de_column is not None:
                    files_info.append((file_path, de_column, fault_type))
                    print(f"Loaded: {mat_file} -> {de_column} -> {fault_type} ({self.args.sampling_rate}Hz)")
                else:
                    print(f"Warning: No DE column found in '{mat_file}', skipping...")

            except Exception as e:
                print(f"Error reading '{mat_file}': {e}")
                continue

        return files_info

    def _identify_fault_type(self, filename):
        """
        根据文件名识别故障类型

        Args:
            filename: 文件名

        Returns:
            fault_type: 故障类型字符串，如'normal', 'IR007'等
        """
        filename_lower = filename.lower()

        # 按长度降序排列，避免短模式匹配长模式的问题
        patterns = sorted(self.fault_patterns.keys(), key=len, reverse=True)

        for fault_type in patterns:
            pattern = self.fault_patterns[fault_type]['pattern'].lower()
            if pattern in filename_lower:
                return fault_type

        return None

    def _find_de_column(self, mat_data, filename):
        """
        在mat文件中查找包含'DE'的数据列

        Args:
            mat_data: loadmat返回的数据字典
            filename: 文件名（用于调试）

        Returns:
            de_column: DE数据列名，如果找不到则返回None
        """
        # 查找所有包含'DE'的键
        de_columns = []
        for key in mat_data.keys():
            if 'DE' in key.upper() and not key.startswith('__'):
                # 检查数据形状，确保是时间序列数据
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.size > 1000:  # 至少1000个数据点
                    de_columns.append(key)

        if len(de_columns) == 0:
            return None
        elif len(de_columns) == 1:
            return de_columns[0]
        else:
            # 如果有多个DE列，优先选择包含'time'的
            time_columns = [col for col in de_columns if 'time' in col.lower()]
            if time_columns:
                return time_columns[0]
            else:
                # 选择第一个
                print(f"Multiple DE columns found in '{filename}': {de_columns}, using '{de_columns[0]}'")
                return de_columns[0]

    def _load_all_data(self):
        """
        加载所有数据（用于数据集划分和scaler初始化）

        Returns:
            all_data: 所有数据的numpy array
            all_labels: 所有标签的numpy array
        """
        all_data = []
        all_labels = []

        # CWRU数据集的根路径
        cwru_root_path = os.path.join(self.args.root_path, 'CWRU')

        # 确定要加载的数据源
        data_sources_to_load = []
        if self.args.data_source == 'both':
            data_sources_to_load = ['12k_DE', '48k_DE']
        else:
            data_sources_to_load = [self.args.data_source]

        print(f"Loading data sources: {data_sources_to_load}")
        print(f"Loading workloads: {self.args.workloads}")
        print(f"Using sampling rate: {self.args.sampling_rate}Hz")

        # 加载各数据源和工况数据
        for data_source in data_sources_to_load:
            data_source_path = os.path.join(cwru_root_path, data_source)

            if not os.path.exists(data_source_path):
                print(f"Warning: Data source path '{data_source_path}' does not exist, skipping...")
                continue

            for workload in self.args.workloads:
                # 自动获取文件信息
                files_info = self._get_workload_files(data_source_path, workload)

                if not files_info:
                    print(f"Warning: No valid files found in {data_source}/{workload}, skipping...")
                    continue

                # 处理每个文件
                for file_path, data_column, fault_type in files_info:
                    try:
                        # 读取mat文件
                        mat_data = loadmat(file_path)
                        signal_data = mat_data[data_column].reshape(-1)

                        print(f"Original data length for {os.path.basename(file_path)}: {len(signal_data)}")

                        # 根据采样率调整数据长度
                        if self.args.sampling_rate == 48000:
                            # 48k数据，使用更多数据点
                            if fault_type == 'normal':
                                signal_data = signal_data[:480000] if len(signal_data) >= 480000 else signal_data
                            else:
                                signal_data = signal_data[:240000] if len(signal_data) >= 240000 else signal_data
                        else:
                            # 12k数据，保持原来的逻辑
                            if fault_type == 'normal':
                                signal_data = signal_data[:240000] if len(signal_data) >= 240000 else signal_data
                            else:
                                signal_data = signal_data[:119808]

                        # 滑动窗口采样
                        windows = self._sliding_window(signal_data)

                        # 添加数据
                        all_data.extend(windows)

                        # 根据任务类型设置标签
                        fault_config = self.fault_patterns[fault_type]
                        if self.args.task_type == '4class':
                            label = fault_config['class_4']
                        else:
                            label = fault_config['class_10']

                        all_labels.extend([label] * len(windows))

                        print(f"Processed {os.path.basename(file_path)}: {len(windows)} windows, "
                              f"label: {label}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

        # 转换为numpy数组
        if len(all_data) == 0:
            raise ValueError(f"No data loaded for data_source '{self.args.data_source}' "
                             f"and workloads '{self.args.workloads}'. Please check your data path.")

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
              f"{self.args.task_type} ({self.num_classes} classes)")

        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

    def _sliding_window(self, signal_data):
        """滑动窗口采样"""
        windows = []
        for i in range(0, len(signal_data) - self.args.window_size + 1, self.args.stride):
            window = signal_data[i:i + self.args.window_size]
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