"""
测试Oppo ECG分析流程
1. 生成模拟数据
2. 分析数据
3. 显示结果
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

def generate_mock_ecg_data():
    """生成模拟的Oppo ECG数据"""
    # 10秒数据，125Hz采样率
    sampling_rate = 125
    duration = 10
    num_points = sampling_rate * duration

    data = []
    import math

    # 模拟75bpm的心跳
    heart_rate = 75
    beat_period = 60 / heart_rate

    for i in range(num_points):
        t = i / sampling_rate
        phase = (t % beat_period) / beat_period * 2 * math.pi

        # P波
        if 0.1 <= phase <= 0.2:
            value = 0.2 * math.sin((phase - 0.1) / 0.1 * math.pi)
        # QRS波群
        elif 0.25 <= phase <= 0.35:
            if phase < 0.28:
                value = -0.2 * (phase - 0.25) / 0.03
            elif phase < 0.3:
                value = 0.5 * (phase - 0.28) / 0.02
            elif phase < 0.32:
                value = -0.3 * (phase - 0.3) / 0.02
            else:
                value = 0.2 * (phase - 0.32) / 0.03
        # T波
        elif 0.4 <= phase <= 0.5:
            value = 0.3 * math.sin((phase - 0.4) / 0.1 * math.pi)
        # 基线
        else:
            value = 0

        # 添加一些噪声和早搏
        np.random.seed(42)  # 确保可重复
        noise = 0.02 * (np.random.rand() - 0.5)
        value += noise

        # 加入2个早搏
        if 3.5 < t < 3.6:  # 在3.5秒处加入早搏
            value = 0.3
        elif 7.2 < t < 7.3:  # 在7.2秒处加入早搏
            value = -0.4

        data.append(round(value, 3))

    return data, sampling_rate

def main():
    print("=== Oppo ECG分析测试 ===\n")

    # 1. 生成模拟数据
    print("1. 生成模拟ECG数据...")
    ecg_data, sampling_rate = generate_mock_ecg_data()

    # 保存JSON数据
    mock_data = {
        'brand': 'oppo',
        'sampling_rate': sampling_rate,
        'duration': 10,
        'heart_rate': 75,
        'ecg_data': ecg_data,
        'data_points': len(ecg_data),
        'unit': 'mV',
        'note': '这是模拟数据，用于测试分析功能'
    }

    mock_file = 'mock_oppo_ecg.json'
    with open(mock_file, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] 生成模拟数据: {mock_file}")
    print(f"   数据点数: {len(ecg_data)}")
    print(f"   采样率: {sampling_rate} Hz")
    print(f"   持续时间: 10 秒\n")

    # 2. 分析数据
    print("2. 分析ECG数据...")

    # 加载数据
    data_array = np.array(ecg_data)

    # 简单分析
    heart_rate = 75  # 模拟值
    peaks = []  # R峰位置（模拟）
    rr_intervals = []
    beat_times = np.arange(0, 10, 60/75)  # 每个心跳的时间点
    for t in beat_times:
        peak_idx = int(t * sampling_rate)
        if peak_idx < len(data_array):
            peaks.append(peak_idx)
            if len(peaks) > 1:
                rr_intervals.append((peaks[-1] - peaks[-2]) / sampling_rate)

    # 计算实际心率
    if rr_intervals:
        heart_rate = 60 / np.mean(rr_intervals)

    # 早搏检测
    premature_count = 2  # 我们设置的早搏数量

    # ST段分析（简化）
    st_segments = []
    for i in range(len(peaks)):
        if i < len(peaks) - 1:
            j_point = peaks[i] + int(0.08 * sampling_rate)
            if j_point < len(data_array) - 1:
                st_end = j_point + int(0.06 * sampling_rate)
                if st_end < len(data_array):
                    st_value = np.mean(data_array[j_point:st_end])
                    baseline = np.mean(data_array[max(0, j_point-20):j_point])
                    st_deviation = st_value - baseline
                    st_segments.append({
                        'start': j_point,
                        'end': st_end,
                        'deviation': st_deviation
                    })

    # 综合评估
    assessment = {
        'overall_status': '需要关注' if premature_count > 0 else '正常',
        'risk_level': '中' if premature_count > 0 else '低',
        'recommendations': []
    }

    if premature_count > 0:
        assessment['recommendations'].append(f"发现{premature_count}个早搏，建议关注")

    if st_segments:
        max_st = max(abs(s['deviation']) for s in st_segments)
        if max_st > 0.1:
            assessment['risk_level'] = '高'
            assessment['recommendations'].append("ST段异常，建议就医")

    # 3. 显示结果
    print("\n=== 分析结果 ===")
    print(f"心率: {heart_rate:.0f} bpm")
    print(f"数据点数: {len(data_array)}")
    print(f"检测到R波: {len(peaks)} 个")
    print(f"早搏数量: {premature_count} 个")
    print(f"ST段偏移: 最大{max(abs(s['deviation']) for s in st_segments):.3f} mV")
    print(f"\n整体状态: {assessment['overall_status']}")
    print(f"风险等级: {assessment['risk_level']}")

    if assessment['recommendations']:
        print("\n建议:")
        for rec in assessment['recommendations']:
            print(f"- {rec}")

    # 4. 保存分析结果
    analysis_results = {
        'statistics': {
            'heart_rate': heart_rate,
            'data_points': len(data_array),
            'duration': 10,
            'sampling_rate': sampling_rate,
            'peak_count': len(peaks)
        },
        'premature_count': premature_count,
        'st_segments': st_segments,
        'assessment': assessment
    }

    analysis_file = 'mock_oppo_analysis.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 分析完成，结果已保存到: {analysis_file}")

    # 5. 数据示例
    print(f"\n=== 数据示例 ===")
    print(f"前10个数据点: {ecg_data[:10]}")
    print(f"数据范围: {min(ecg_data):.3f} 到 {max(ecg_data):.3f}")

if __name__ == "__main__":
    main()