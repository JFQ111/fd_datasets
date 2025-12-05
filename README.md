# 轴承故障诊断数据集加载器

## 项目概述

基于 PyTorch 的轴承故障诊断数据集加载器，覆盖多种公开数据集，统一完成滑窗、标准化、特征变换与数据划分。支持多数据集混合训练、缓存复用与加权采样，适用于二分类、多分类和多域场景。

## 核心特性

- 多数据集：CWRU、PU、JNU、MFPT、HUST、HUST_V，可按需混合加载。
- 统一预处理：滑窗采样、标准化，可选 CWT/STFT/GAF/RP/Scalogram 等变换并调整目标尺寸。
- 高效数据管线：Dataset/DataLoader 封装，支持多进程加载与加权随机采样。
- 缓存机制：按参数生成哈希键，将 train/val/test 拆分结果缓存到 `./datasets/cache`，避免重复预处理。
- 域标签：多数据集组合时自动为样本添加 `domain` 字段，便于多域/迁移学习。
- 灵活任务：支持 4/10/13/15 等多种分类设定，振动/电流/融合信号均可。

## 支持的数据集与任务

- **CWRU**：4 类或 10 类；12k_DE / 48k_DE / both。
- **PU**：3/5/9/13 人工故障，15 类自然故障；振动、电流或二者融合。
- **JNU**：4 类（正常、内圈、外圈、滚动体），常用转速 600/800/1000。
- **MFPT**：3 类（正常、内圈、外圈），自动重采样到 48828Hz。
- **HUST**：4 类或 9 类；支持 x/y/z/xyz 四种信号组合，工况从文件名提取。
- **HUST_V**：4 类或 7 类；轴承 6204-6208，工况 0/2/4（对应 0W/200W/400W），采样率固定 51200Hz。

## 数据目录示例

- 默认根目录：`./datasets`
- 典型结构：
  - `CWRU/12k_DE/0hp/*.mat`
  - `PU/<bearing_id>/*.mat`（文件名包含工况，如 `N09_M07_F10_*.mat`）
  - `JNU/*.csv`（文件名含故障与转速，如 `tb1000_2.csv`）
  - `MFPT/1 - Three Baseline Conditions/*.mat` 等官方目录
  - `HUST/*.xlsx`（第 3/4/5 列为 x/y/z，加速度数据从第 23 行开始）
  - `HUST_V/<bearing>/<workload>/*.mat`
  - `cache/`（运行后自动生成的缓存）

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

1. 下载并解压数据到 `./datasets`：
   - [Datasets](https://pan.quark.cn/s/71183ea35905)（提取码：sY4q）
   - HUST/HUST_V：夸克网盘 [HUST+HUST_V.zip](https://pan.quark.cn/s/6f3a5a4ed893)（提取码：ysBb，将其放在datasets文件夹下）
2. 运行示例脚本（示例脚本的命令行目前覆盖 CWRU/PU/JNU/MFPT）：
   ```bash
   # 默认 CWRU，4 类，12k_DE
   python example.py --dataset CWRU --root_path ./datasets

   # 多数据集 + 小波变换
   python example.py --dataset CWRU PU --transform_type cwt --cwt_scales 64

   # JNU 指定转速与窗口
   python example.py --dataset JNU --jnu_workloads 600 1000 --window_size 512 --stride 256

   # 多域训练示例（两数据集不同权重）
   python example.py --dataset CWRU MFPT --dataset_weights CWRU:1.0,MFPT:0.7
   ```
3. 如果需要直接使用 HUST/HUST_V，请在 `example.py` 基础上补充对应参数（见下文“参数速查”）或在自定义脚本中构造 `argparse.Namespace` 后调用 `dataprovider.create_dataloaders(args)`。

## 参数速查

- **通用/滑窗与划分**
  - `--dataset`：数据集列表，如 `CWRU PU MFPT`。
  - `--dataset_weights`：多数据集权重，格式 `CWRU:1.0,PU:0.8`，用于 WeightedRandomSampler。
  - `--root_path`：数据集根目录，默认 `./datasets`。
  - `--sampling_rate`：采样率（部分数据集会覆盖为固定值）。
  - `--window_size` / `--stride`：滑动窗口大小与步长。
  - `--train_ratio` / `--val_ratio`：训练/验证集比例，其余为测试集。
- **预处理与特征变换**
  - `--normalize`：是否使用训练集拟合标准化。
  - `--transform_type`：`None`/`cwt`/`stft`/`gaf`/`rp`/`scalogram`。
  - `--target_size`：二维变换后目标尺寸，默认 `(32, 32)`。
  - `--cwt_scales`，`--stft_nperseg`，`--stft_noverlap`，`--gaf_method`，`--rp_eps`：对应变换的细节参数。
- **DataLoader**
  - `--batch_size`，`--num_workers`，`--pin_memory`，`--shuffle`，`--drop_last`。
- **CWRU**
  - `--data_source`：`12k_DE`/`48k_DE`/`both`。
  - `--workloads`：`0hp` `1hp` `2hp` `3hp`。
  - `--task_type`：`4class` 或 `10class`。
- **PU**
  - `--pu_workloads`：工况筛选，如 `N15_M07_F10`。
  - `--pu_task_type`：`3class_artificial` / `5class_artificial` / `9class_artificial` / `13class_artificial` / `15class_nature`。
  - `--pu_signal_type`：`vibration` / `current` / `both`。
- **JNU**
  - `--jnu_workloads`：转速列表，如 `600 800 1000`。
- **HUST**
  - `--hust_workloads`：`all` 或文件名中包含的工况（如 `2000Hz`、`VS`），默认 `all`。
  - `--hust_task_type`：`4class`（忽略复合故障）或 `9class`。
  - `--hust_signal_type`：`x` / `y` / `z` / `xyz`（三向合成模长）。
- **HUST_V**
  - `--hustv_bearings`：轴承型号列表，如 `6204 6205 6206`。
  - `--hustv_workloads`：`0` `2` `4`。
  - `--hustv_task_type`：`4class`（忽略复合故障）或 `7class`。

## 缓存与采样

- 数据缓存存放于 `./datasets/cache/datasets_<hash>.pkl`，由所有参数共同决定；调整参数后会自动生成新缓存。
- 需要清理缓存可调用：
  ```python
  from dataprovider.data_factory import clear_cache
  clear_cache('./datasets/cache')          # 全部清理
  clear_cache('./datasets/cache', '*.pkl') # 按模式清理
  ```
- 当提供 `--dataset_weights` 时，训练 DataLoader 将使用 `WeightedRandomSampler` 以控制各域采样比例；每个样本自动附带 `domain` 字段便于多域训练。

## 贡献指南

- 提交 Issue 反馈问题或建议。
- 提交 Pull Request 合并代码。
- 协助文档撰写与完善。

请确保遵循代码规范，并提供必要的测试代码。

## 联系方式

2410392@tongji.edu.cn

## 鸣谢

感谢 Claude 的支持
