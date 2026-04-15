"""
ECG 信号处理器
用于ECG信号的预处理、去噪和增强
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ECGSignalProcessor:
    """ECG信号处理器"""

    def __init__(self, sampling_rate: int = 128):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2

    def preprocess(self, ecg_data: np.ndarray) -> np.ndarray:
        """ECG信号预处理流程"""
        # 1. 去除基线漂移
        filtered = self._remove_baseline_drift(ecg_data)

        # 2. 带通滤波
        filtered = self._bandpass_filter(filtered)

        # 3. 工频干扰滤波（50/60Hz）
        filtered = self._notch_filter(filtered)

        # 4. 平滑处理
        filtered = self._smooth_signal(filtered)

        return filtered

    def _remove_baseline_drift(self, ecg_data: np.ndarray) -> np.ndarray:
        """去除基线漂移"""
        # 使用高通滤波
        highpass_cutoff = 0.5  # Hz
        b, a = butter(4, highpass_cutoff / self.nyquist, btype='high')
        return filtfilt(b, a, ecg_data)

    def _bandpass_filter(self, ecg_data: np.ndarray) -> np.ndarray:
        """带通滤波（0.5-40Hz）"""
        low_cutoff = 0.5  # Hz
        high_cutoff = 40  # Hz

        b, a = butter(4, [low_cutoff / self.nyquist, high_cutoff / self.nyquist], btype='band')
        return filtfilt(b, a, ecg_data)

    def _notch_filter(self, ecg_data: np.ndarray) -> np.ndarray:
        """陷波滤波去除工频干扰"""
        # 假设是50Hz干扰（可根据实际情况调整）
        freq = 50
        Q = 30  # 品质因数

        b, a = butter(4, freq / self.nyquist, btype='band')
        return filtfilt(b, a, ecg_data)

    def _smooth_signal(self, ecg_data: np.ndarray) -> np.ndarray:
        """信号平滑"""
        # 使用移动平均
        window_size = int(0.02 * self.sampling_rate)  # 20ms窗口
        if window_size < 3:
            window_size = 3

        kernel = np.ones(window_size) / window_size
        return np.convolve(ecg_data, kernel, mode='same')

    def detect_r_peaks(self, ecg_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """检测R峰"""
        # 使用Pan-Tompkins算法的简化版本
        # 1. 微分
        diff = np.diff(ecg_data)

        # 2. 平方
        squared = diff ** 2

        # 3. 移动积分
        window_size = int(0.15 * self.sampling_rate)  # 150ms
        integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

        # 4. 检测峰值
        peaks, properties = find_peaks(
            integrated,
            height=np.max(integrated) * 0.3,  # 高度阈值
            distance=int(0.3 * self.sampling_rate)  # 最小间隔300ms
        )

        return peaks, {
            'peak_heights': properties['peak_heights'],
            'widths': properties['widths'],
            'prominences': properties['prominences']
        }

    def calculate_heart_rate(self, r_peaks: np.ndarray, duration: float) -> float:
        """计算心率"""
        if len(r_peaks) < 2:
            return 0

        # 计算RR间期
        rr_intervals = np.diff(r_peaks) / self.sampling_rate

        # 计算心率 (bpm)
        heart_rate = 60 / np.mean(rr_intervals)

        return heart_rate

    def analyze_signal_quality(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """分析信号质量"""
        # 计算信噪比
        signal_power = np.mean(processed ** 2)
        noise = original - processed
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            snr = float('inf')
        else:
            snr = 10 * np.log10(signal_power / noise_power)

        # 评估信号质量
        if snr > 20:
            quality = "优秀"
        elif snr > 15:
            quality = "良好"
        elif snr > 10:
            quality = "一般"
        else:
            quality = "较差"

        return {
            'snr_db': snr,
            'quality': quality,
            'signal_power': signal_power,
            'noise_power': noise_power
        }


def process_ecg_signal(ecg_data: np.ndarray, sampling_rate: int = 128) -> Dict:
    """处理ECG信号的便捷函数"""
    processor = ECGSignalProcessor(sampling_rate)

    # 预处理
    processed = processor.preprocess(ecg_data)

    # 检测R峰
    r_peaks, peak_info = processor.detect_r_peaks(processed)

    # 计算心率（假设10秒数据）
    duration = 10  # 秒
    heart_rate = processor.calculate_heart_rate(r_peaks, duration)

    # 分析信号质量
    quality = processor.analyze_signal_quality(ecg_data, processed)

    return {
        'original_signal': ecg_data,
        'processed_signal': processed,
        'r_peaks': r_peaks,
        'heart_rate': heart_rate,
        'peak_info': peak_info,
        'signal_quality': quality,
        'sampling_rate': sampling_rate
    }