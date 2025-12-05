import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from transforms.signal_transforms import apply_transform


class JNUBearingDataset(Dataset):
    """
    JNU Bearing Fault Diagnosis Dataset
    江南大学轴承故障诊断数据集
    支持4分类任务（正常、内圈故障、外圈故障、滚动体故障）
    """

    def __init__(self, args, flag='train'):
        """
        Args:
            args: 命令行参数对象
            flag: 'train', 'val', 'test'
        """
        # args.stride = 128
        self.args = args
        self.args.sampling_rate = 50000  # JNU数据集默认采样率
        self.flag = flag

        # 初始化标准化器
        if args.normalize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # 定义类别映射 - 基于文件夹名称识别
        self.fault_patterns = {
            'n': {'pattern': 'n', 'class': 0, 'name': 'normal'},  # 正常状态
            'ib': {'pattern': 'ib', 'class': 1, 'name': 'inner'},  # 内圈故障
            'ob': {'pattern': 'ob', 'class': 2, 'name': 'outer'},  # 外圈故障
            'tb': {'pattern': 'tb', 'class': 3, 'name': 'ball'},  # 滚动体故障
        }

        # 定义类别名称
        self.class_names = ['normal', 'inner', 'outer', 'ball']
        self.num_classes = 4
        # 数据集数据和标签
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

    def _get_workload_files(self, jnu_root_path, workload):
        """
        获取指定工况下的所有CSV文件
        通过文件名识别故障类型和转速，如tb1000_2.csv表示1000转速滚动体故障

        Args:
            jnu_root_path: JNU数据集根路径
            workload: 工况名称（如 '600', '800', '1000'）

        Returns:
            files_info: [(file_path, fault_type), ...]
        """
        files_info = []

        if not os.path.exists(jnu_root_path):
            print(f"Warning: JNU root path '{jnu_root_path}' does not exist.")
            return files_info

        # 获取JNU文件夹下所有CSV文件
        all_files = [f for f in os.listdir(jnu_root_path) if f.endswith('.csv')]
        all_files.sort()  # 保证顺序一致性

        print(f"Found {len(all_files)} total CSV files in {jnu_root_path}")

        # 统计各故障类型的文件数量
        fault_counts = {fault_key: 0 for fault_key in self.fault_patterns.keys()}

        # 遍历所有CSV文件，匹配工况和故障类型
        for csv_file in all_files:
            # 解析文件名，获取故障类型和转速
            fault_type, file_workload = self._parse_filename(csv_file)

            if fault_type is None:
                print(f"Warning: Cannot identify fault type from filename '{csv_file}', skipping...")
                continue

            if file_workload != workload:
                continue  # 跳过不匹配的工况

            # 添加文件信息
            file_path = os.path.join(jnu_root_path, csv_file)
            files_info.append((file_path, fault_type))
            fault_counts[fault_type] += 1

            print(f"Added: {csv_file} -> {self.fault_patterns[fault_type]['name']} fault at {workload} rpm")

        # 打印各故障类型的文件统计
        print(f"Files summary for {workload} rpm:")
        for fault_key, count in fault_counts.items():
            if count > 0:
                print(f"  {self.fault_patterns[fault_key]['name']}: {count} files")

        return files_info

    def _parse_filename(self, filename):
        """
        解析文件名，提取故障类型和转速

        Args:
            filename: 文件名，如 'tb1000_2.csv', 'n600_3_2.csv', 'ib800_2.csv'

        Returns:
            fault_type: 故障类型 ('n', 'ib', 'ob', 'tb')
            workload: 转速字符串 ('600', '800', '1000')
        """
        filename_lower = filename.lower().replace('.csv', '')

        # 按长度降序排列故障类型，避免短模式匹配长模式的问题
        fault_types = sorted(self.fault_patterns.keys(), key=len, reverse=True)

        for fault_type in fault_types:
            if filename_lower.startswith(fault_type):
                # 提取故障类型后面的数字部分作为转速
                remaining = filename_lower[len(fault_type):]

                # 提取转速数字（通常在开头）
                workload = ''
                for char in remaining:
                    if char.isdigit():
                        workload += char
                    else:
                        break  # 遇到非数字字符就停止

                if workload:
                    return fault_type, workload
                else:
                    print(f"Warning: No workload found in filename '{filename}'")
                    return None, None

        return None, None

    def _load_csv_data(self, file_path):
        """
        加载CSV文件数据

        Args:
            file_path: CSV文件路径

        Returns:
            signal_data: 信号数据数组
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 假设信号数据在第一列，如果数据结构不同可以调整
            if len(df.columns) >= 1:
                signal_data = df.iloc[:, 0].values  # 取第一列作为信号数据
            else:
                raise ValueError(f"CSV file {file_path} has no data columns")

            # 确保数据类型为float
            signal_data = signal_data.astype(np.float32)

            return signal_data

        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return None

    def _load_all_data(self):
        """
        加载所有数据（用于数据集划分和scaler初始化）

        Returns:
            all_data: 所有数据的numpy array
            all_labels: 所有标签的numpy array
        """
        all_data = []
        all_labels = []

        # JNU数据集的根路径
        jnu_root_path = os.path.join(self.args.root_path, 'JNU')

        if not os.path.exists(jnu_root_path):
            raise ValueError(f"JNU dataset path '{jnu_root_path}' does not exist. Please check your root_path.")

        print(f"Loading JNU dataset from: {jnu_root_path}")
        print(f"Loading workloads: {self.args.jnu_workloads}")

        # 加载各工况数据
        for workload in self.args.jnu_workloads:
            print(f"\nProcessing workload: {workload} rpm")

            # 获取当前工况的所有文件
            files_info = self._get_workload_files(jnu_root_path, workload)

            if not files_info:
                print(f"Warning: No valid files found for workload {workload}, skipping...")
                continue

            # 处理每个文件
            for file_path, fault_type in files_info:
                # 读取CSV数据
                signal_data = self._load_csv_data(file_path)

                if signal_data is None:
                    continue

                print(f"Original data length for {os.path.basename(file_path)}: {len(signal_data)}")

                # 滑动窗口采样
                windows = self._sliding_window(signal_data)

                if len(windows) == 0:
                    print(f"Warning: No windows generated from {os.path.basename(file_path)}")
                    continue

                # 添加数据
                all_data.extend(windows)

                # 设置标签
                label = self.fault_patterns[fault_type]['class']
                all_labels.extend([label] * len(windows))

                print(f"Processed {os.path.basename(file_path)}: {len(windows)} windows, "
                      f"label: {label} ({self.fault_patterns[fault_type]['name']})")

        # 转换为numpy数组
        if len(all_data) == 0:
            print("Available CSV files in JNU dataset:")
            jnu_root_path = os.path.join(self.args.root_path, 'JNU')
            if os.path.exists(jnu_root_path):
                csv_files = [f for f in os.listdir(jnu_root_path) if f.endswith('.csv')]
                for csv_file in csv_files[:10]:  # 显示前10个文件作为示例
                    fault_type, workload = self._parse_filename(csv_file)
                    print(f"  {csv_file} -> fault: {fault_type}, workload: {workload}")
                if len(csv_files) > 10:
                    print(f"  ... and {len(csv_files) - 10} more files")

            raise ValueError(f"No data loaded for workloads '{self.args.jnu_workloads}'. "
                             f"Please check your data path and workload settings. "
                             f"Make sure CSV files follow naming pattern like 'tb1000_2.csv'.")

        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        print(f"\nTotal loaded: {len(all_data)} samples")

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

        print(f"\nLoaded {self.flag} dataset: {len(self.data)} samples, 4 classes")

        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = {}
        for class_id, count in zip(unique, counts):
            class_name = self.class_names[class_id]
            class_distribution[f"{class_id}({class_name})"] = count
        print(f"Class distribution: {class_distribution}")

    def _sliding_window(self, signal_data):
        """滑动窗口采样"""
        windows = []

        if len(signal_data) < self.args.window_size:
            print(f"Warning: Signal length {len(signal_data)} < window size {self.args.window_size}")
            return windows

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