import numpy as np
from scipy import signal
import pywt
from scipy.ndimage import zoom


class SignalTransforms:
    """信号变换工具类"""

    @staticmethod
    def resize_to_target_size(data, target_size=(32, 32)):
        """调整数据大小到目标尺寸"""
        if data.shape != target_size:
            zoom_factor = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])
            data = zoom(data, zoom_factor, order=3)  # Using cubic interpolation for smooth resizing
        return data

    @staticmethod
    def continuous_wavelet_transform(signal_window, sampling_rate, scales=64, target_size=(32, 32)):
        """连续小波变换"""
        scales = np.arange(1, scales + 1)
        coefficients, _ = pywt.cwt(signal_window, scales, 'morl', sampling_period=1 / sampling_rate)
        return SignalTransforms.resize_to_target_size(np.abs(coefficients) ** 2, target_size)

    @staticmethod
    def short_time_fourier_transform(signal_window, sampling_rate, nperseg=64, noverlap=32, target_size=(32, 32)):
        """短时傅里叶变换"""
        f, t, Zxx = signal.stft(signal_window, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        return SignalTransforms.resize_to_target_size(np.abs(Zxx), target_size)

    @staticmethod
    def gramian_angular_field(signal_window, method='summation', target_size=(32, 32)):
        """格拉姆角场变换"""
        # 归一化到[-1, 1]
        min_val = np.min(signal_window)
        max_val = np.max(signal_window)
        if max_val == min_val:
            scaled = np.zeros_like(signal_window)
        else:
            scaled = 2 * (signal_window - min_val) / (max_val - min_val) - 1

        # 转换为极坐标
        # 处理数值溢出
        scaled = np.clip(scaled, -1, 1)
        phi = np.arccos(scaled)

        # 计算格拉姆矩阵
        if method == 'summation':
            gaf = np.cos(phi[:, None] + phi[None, :])
        else:
            gaf = np.sin(phi[:, None] - phi[None, :])

        # 如果窗口太大，进行降采样
        return SignalTransforms.resize_to_target_size(gaf, target_size)

    @staticmethod
    def recurrence_plot(signal_window, eps=0.1, target_size=(32, 32)):
        """递归图"""
        N = len(signal_window)
        rp = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                rp[i, j] = 1 if np.abs(signal_window[i] - signal_window[j]) < eps else 0

        # 如果窗口太大，进行降采样
        return SignalTransforms.resize_to_target_size(rp, target_size)

    @staticmethod
    def scalogram(signal_window, sampling_rate, scales=64, target_size=(32, 32)):
        """尺度图"""
        scales = np.logspace(0, 2, scales)
        coefficients, _ = pywt.cwt(signal_window, scales, 'morl', sampling_period=1 / sampling_rate)
        return SignalTransforms.resize_to_target_size(np.abs(coefficients) ** 2, target_size)


def apply_transform(signal_window, sampling_rate, args):
    """应用信号变换"""
    target_size = getattr(args, 'target_size', (32, 32))  # Default target_size to (32, 32) if not provided

    if args.transform_type == 'None':
        return signal_window

    elif args.transform_type == 'cwt':
        return SignalTransforms.continuous_wavelet_transform(
            signal_window, sampling_rate, args.cwt_scales, target_size)

    elif args.transform_type == 'stft':
        return SignalTransforms.short_time_fourier_transform(
            signal_window, sampling_rate, args.stft_nperseg, args.stft_noverlap, target_size)

    elif args.transform_type == 'gaf':
        return SignalTransforms.gramian_angular_field(
            signal_window, args.gaf_method, target_size)

    elif args.transform_type == 'rp':
        return SignalTransforms.recurrence_plot(
            signal_window, args.rp_eps, target_size)

    elif args.transform_type == 'scalogram':
        return SignalTransforms.scalogram(
            signal_window, sampling_rate, args.cwt_scales, target_size)

    else:
        raise ValueError(f"Unknown transform type: {args.transform_type}")

