"""
ECG 分析器
用于分析ECG数据，检测心脏健康指标
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ECGAnalyzer:
    """ECG分析器"""

    def __init__(self, sampling_rate: int = 128):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2

    def analyze_full(self, ecg_data: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """完整的ECG分析"""
        results = {}

        # 1. 基础指标分析
        results.update(self._analyze_basic_metrics(ecg_data, r_peaks))

        # 2. 心律分析
        results.update(self._analyze_rhythm(r_peaks))

        # 3. 早搏检测
        results.update(self._detect_premature_beats(ecg_data, r_peaks))

        # 4. ST段分析
        results.update(self._analyze_st_segment(ecg_data, r_peaks))

        # 5. 综合评估
        results.update(self._overall_assessment(results))

        return results

    def _analyze_basic_metrics(self, ecg_data: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """基础指标分析"""
        if len(r_peaks) < 2:
            return {
                'heart_rate': 0,
                'mean_hr': 0,
                'hr_std': 0,
                'rr_intervals': [],
                'pr_interval': 0,
                'qt_interval': 0,
                'qtc_interval': 0
            }

        # RR间期
        rr_intervals = np.diff(r_peaks) / self.sampling_rate  # 秒
        rr_intervals_ms = rr_intervals * 1000  # 毫秒

        # 心率
        heart_rate = 60 / np.mean(rr_intervals)
        hr_std = np.std(rr_intervals) * 60 / np.mean(rr_intervals)  # HR变异性

        # PR间期 (假设120ms)
        pr_interval = 120  # ms，需要根据实际R波位置调整

        # QT间期 (计算每个心搏的QT)
        qt_intervals = []
        for i in range(len(r_peaks) - 1):
            qt_start = r_peaks[i]
            qt_end = r_peaks[i + 1]
            # 简化估算：QT间期为RR间期的40%
            qt_interval = 0.4 * rr_intervals[i] * 1000  # ms
            qt_intervals.append(qt_interval)

        # QTc间期 (Bazett公式)
        if qt_intervals:
            qt_intervals = np.array(qt_intervals)
            rr_intervals_sec = rr_intervals[:-1]  # 对应的RR间期
            qtc_intervals = qt_intervals / np.sqrt(rr_intervals_sec)
            qtc_interval = np.mean(qtc_intervals)
        else:
            qtc_interval = 0

        return {
            'heart_rate': heart_rate,
            'mean_hr': np.mean(60 / rr_intervals),
            'hr_std': hr_std,
            'rr_intervals': rr_intervals_ms.tolist(),
            'pr_interval': pr_interval,
            'qt_interval': np.mean(qt_intervals) if qt_intervals else 0,
            'qtc_interval': qtc_interval
        }

    def _analyze_rhythm(self, r_peaks: np.ndarray) -> Dict:
        """心律分析"""
        if len(r_peaks) < 10:
            return {
                'rhythm_type': '无法判断',
                'rhythm_regularity': 0,
                'irregular_beats': 0
            }

        # RR间期变异性
        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        rr_var = np.std(rr_intervals) / np.mean(rr_intervals)

        # 心律规则性
        if rr_var < 0.05:
            rhythm_type = "窦性心律"
            rhythm_regularity = "规则"
        elif rr_var < 0.1:
            rhythm_type = "窦性心律不齐"
            rhythm_regularity = "轻度不齐"
        elif rr_var < 0.15:
            rhythm_type = "心律不齐"
            rhythm_regularity = "中度不齐"
        else:
            rhythm_type = "严重心律不齐"
            rhythm_regularity = "显著不齐"

        # 计算不规则搏动数
        # 使用变异系数阈值
        threshold = np.mean(rr_intervals) + 2 * np.std(rr_intervals)
        irregular_beats = np.sum(rr_intervals > threshold)

        return {
            'rhythm_type': rhythm_type,
            'rhythm_regularity': rhythm_regularity,
            'rr_variability': rr_var,
            'irregular_beats': int(irregular_beats)
        }

    def _detect_premature_beats(self, ecg_data: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """早搏检测"""
        if len(r_peaks) < 4:
            return {
                'premature_beats': [],
                'premature_count': 0,
                'premature_type': '无法检测'
            }

        # RR间期
        rr_intervals = np.diff(r_peaks) / self.sampling_rate
        mean_rr = np.mean(rr_intervals)

        # 检测早搏的标准：RR间期突然缩短
        premature_indices = []
        premature_types = []

        for i in range(1, len(rr_intervals) - 1):
            # 当前RR间期比平均RR间期短30%以上
            if rr_intervals[i] < 0.7 * mean_rr:
                # 检查下一个RR间期是否延长（代偿间歇）
                if rr_intervals[i + 1] > 1.2 * mean_rr:
                    premature_indices.append(i)
                    # 根据早搏发生位置判断类型
                    # 简化判断：前1/3为房性，后2/3为室性
                    if i < len(rr_intervals) / 3:
                        premature_types.append("房性早搏")
                    else:
                        premature_types.append("室性早搏")

        premature_count = len(premature_indices)

        return {
            'premature_beats': list(zip(premature_indices, premature_types)),
            'premature_count': premature_count,
            'premature_rate': premature_count / len(rr_intervals) * 100,
            'premature_type': '频繁' if premature_count > 5 else '偶发' if premature_count > 0 else '无'
        }

    def _analyze_st_segment(self, ecg_data: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """ST段分析"""
        if len(r_peaks) < 2:
            return {
                'st_segments': [],
                'st_deviation': 0,
                'st_status': '无法分析'
            }

        st_segments = []
        st_deviations = []

        for i in range(len(r_peaks) - 1):
            # J点（ST段起点）
            j_point = r_peaks[i] + int(0.08 * self.sampling_rate)  # R波后80ms

            # ST段终点（J点后60ms）
            st_end = j_point + int(0.06 * self.sampling_rate)

            # 确保不超出范围
            if st_end >= len(ecg_data):
                continue

            # 测量ST段偏移（相对于基线）
            baseline_value = np.mean(ecg_data[max(0, j_point-20):j_point])
            st_value = np.mean(ecg_data[j_point:st_end])
            st_deviation = st_value - baseline_value

            st_segments.append({
                'start': j_point,
                'end': st_end,
                'deviation': st_deviation,
                'baseline': baseline_value,
                'value': st_value
            })

            st_deviations.append(st_deviation)

        # 计算平均ST段偏移
        mean_st_deviation = np.mean(st_deviations) if st_deviations else 0
        max_st_deviation = np.max(np.abs(st_deviations)) if st_deviations else 0

        # ST段状态判断
        if abs(mean_st_deviation) < 0.05:  # 0.05mV
            st_status = "正常"
        elif -0.1 < mean_st_deviation < 0:
            st_status = "轻度ST段压低"
        elif mean_st_deviation <= -0.1:
            st_status = "ST段压低"
        elif 0 < mean_st_deviation < 0.1:
            st_status = "轻度ST段抬高"
        else:
            st_status = "ST段抬高"

        return {
            'st_segments': st_segments,
            'st_deviation': mean_st_deviation,
            'max_st_deviation': max_st_deviation,
            'st_status': st_status,
            'ischemia_risk': '高' if max_st_deviation > 0.1 else '低'
        }

    def _overall_assessment(self, results: Dict) -> Dict:
        """综合评估"""
        assessment = {
            'overall_status': '正常',
            'risk_level': '低',
            'recommendations': []
        }

        # 基于心率评估
        hr = results.get('heart_rate', 0)
        if hr < 50 or hr > 100:
            assessment['overall_status'] = '需要关注'
            if hr < 50:
                assessment['recommendations'].append("心率偏慢，建议咨询医生")
            else:
                assessment['recommendations'].append("心率偏快，建议咨询医生")

        # 基于早搏评估
        premature_count = results.get('premature_count', 0)
        if premature_count > 10:
            assessment['overall_status'] = '需要关注'
            assessment['risk_level'] = '中'
            assessment['recommendations'].append("发现较多早搏，建议进一步检查")

        # 基于ST段评估
        st_status = results.get('st_status', '正常')
        if '压低' in st_status or '抬高' in st_status:
            assessment['overall_status'] = '需要关注'
            assessment['risk_level'] = '高'
            assessment['recommendations'].append("ST段异常，建议立即就医")

        # 心律评估
        rhythm_type = results.get('rhythm_type', '无法判断')
        if '严重' in rhythm_type:
            assessment['overall_status'] = '需要关注'
            assessment['risk_level'] = '中'
            assessment['recommendations'].append("心律不齐明显，建议咨询医生")

        return assessment


def analyze_ecg(ecg_data: np.ndarray, r_peaks: np.ndarray, sampling_rate: int = 128) -> Dict:
    """ECG分析的便捷函数"""
    analyzer = ECGAnalyzer(sampling_rate)
    return analyzer.analyze_full(ecg_data, r_peaks)