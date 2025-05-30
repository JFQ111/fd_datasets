# 轴承故障诊断数据集加载器

## 项目概述

本项目提供了一个完整的轴承故障诊断数据集加载器解决方案，支持多种主流轴承故障数据集（CWRU、PU、JNU、MFPT），具备如下特点：

- **多数据集支持**：可加载不同来源、不同格式的数据集，统一预处理与特征提取。
- **灵活预处理选项**：支持多种信号变换方法，如小波变换、短时傅里叶变换、格拉姆角场等。
- **数据集划分管理**：支持训练集、验证集和测试集的自定义划分及标准化处理。
- **高效数据加载机制**：基于 PyTorch 的 Dataset 和 DataLoader 类，提升批处理效率。
- **缓存机制**：避免重复预处理，提升加载效率。
- **多任务支持**：支持二分类、多分类等多种故障识别任务。

## 支持的数据集

- **CWRU 数据集**：
  - 支持 4 分类与 10 分类任务。
  - 支持 12k 和 48k 两种采样率。
- **PU 数据集**：
  - 支持 3、5、9、13（人工故障） 分类任务以及15分类（自然）任务。
  - 支持振动、电流或两者融合信号。
- **JNU 数据集**：
  - 支持 4 分类任务。
- **MFPT 数据集**：
  - 支持 3 分类任务（正常、内圈、外圈故障）。

## 数据预处理与特征提取

- 支持信号转换方式：
  - 连续小波变换（CWT）
  - 短时傅里叶变换（STFT）
  - 格拉姆角场（GAF）
  - 递归图（RP）
  - 尺度图（Scalogram）
- 滑动窗口采样：自定义窗口大小与步长。
- 数据标准化选项：可选择是否使用训练集进行归一化初始化。

## 安装说明

### 环境依赖

- Python 3.6+
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Pandas
- Matplotlib

### 安装命令

```bash
pip install -r requirements.txt
```

## 快速开始

1. 下载并解压数据集到 `./datasets` 目录：

   - [Datasets](https://pan.quark.cn/s/71183ea35905)
   - 提取码：sY4q

2. 运行示例脚本：

```bash
python example.py --dataset CWRU --root_path ./datasets
```

## 参数说明

### 通用参数

```plaintext
--dataset: 数据集类型（CWRU, PU, JNU, MFPT）
--root_path: 数据集根目录（默认 ./datasets）
```

### CWRU 参数

```plaintext
--data_source: 采样率（12k_DE, 48k_DE, both）
--workloads: 工况（0hp, 1hp, 2hp, 3hp）
--task_type: 分类任务类型（4class 或 10class）
```

### PU 参数

```plaintext
--pu_workloads: 工况选择
--pu_task_type: 分类任务类型
--pu_signal_type: 信号类型（vibration, current, both）
```

### JNU 参数

```plaintext
--jnu_workloads: 转速选择（如 600, 800, 1000）
```

### 滑窗与划分参数

```plaintext
--sampling_rate: 采样率（Hz）
--window_size: 滑动窗口大小
--stride: 滑动步长
--train_ratio: 训练集比例
--val_ratio: 验证集比例
```

### 数据预处理参数

```plaintext
--normalize: 是否标准化（默认 True）
--transform_type: 数据变换类型（None, cwt, stft, gaf, rp, scalogram）
```

### DataLoader 参数

```plaintext
--batch_size: 批次大小
--num_workers: 加载线程数
--pin_memory: 是否使用 pin_memory
--shuffle: 是否打乱数据
--drop_last: 是否丢弃最后不完整批
```

## 示例用法

### 默认加载 CWRU：

```bash
python example.py --dataset CWRU 
```

### 加载多个数据集并使用小波变换：

```bash
python example.py --dataset CWRU PU  --transform_type cwt --cwt_scales 64
```

### 加载 JNU 数据集并指定工况和窗口：

```bash
python example.py --dataset JNU --root_path ./datasets --jnu_workloads 600 1000 --window_size 512 --stride 256
```
## 贡献指南

欢迎为本项目贡献代码与改进建议：

1. 提交 Issue 反馈问题或建议
2. 提交 Pull Request 合并代码
3. 协助文档撰写与完善

请确保遵循代码规范，并提供必要的测试代码。

## 联系方式
如有任何问题或建议，请通过以下方式联系：
2410392@tongji.edu.cn

## 鸣谢
感谢Claude的支持