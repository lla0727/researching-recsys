"""
ECG数据分析工具
直接分析提供的数据
"""

import json
import sys
import numpy as np
from pathlib import Path

def load_ecg_data(file_path):
    """加载ECG数据"""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['ecg_data'], data.get('sampling_rate', 125)
    elif file_path.endswith('.csv'):
        # 假设CSV只有一列ecg_value
        import pandas as pd
        df = pd.read_csv(file_path)
        return df['ecg_value'].values.tolist(), 125
    else:
        # 尝试解析为数值数组
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 提取所有数字
            numbers = re.findall(r'[+-]?\d+\.?\d*', content)
            return [float(n) for n in numbers], 125
        except:
            print(f"无法解析文件: {file_path}")
            sys.exit(1)

def analyze_ecg(ecg_data, sampling_rate):
    """分析ECG数据"""
    ecg_array = np.array(ecg_data)

    # 1. 信号预处理
    # 简单的去噪
    from scipy.signal import butter, filtfilt

    # 高通滤波去除基线漂移
    nyquist = sampling_rate / 2
    low_cutoff = 0.5
    high_cutoff = 40

    b, a = butter(4, [low_cutoff/nyquist, high_cutoff/nyquist], btype='band')
    filtered = filtfilt(b, a, ecg_array)

    # 2. 检测R峰
    from scipy.signal import find_peaks

    # 简单的峰值检测
    peaks, properties = find_peaks(
        filtered,
        height=np.max(filtered) * 0.3,
        distance=int(0.3 * sampling_rate)  # 最小300ms间隔
    )

    # 3. 计算心率
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate
        heart_rate = 60 / np.mean(rr_intervals)
    else:
        heart_rate = 0

    # 4. 基础统计
    stats = {
        'heart_rate': heart_rate,
        'mean_value': np.mean(ecg_array),
        'std_value': np.std(ecg_array),
        'min_value': np.min(ecg_array),
        'max_value': np.max(ecg_array),
        'duration': len(ecg_array) / sampling_rate,
        'sampling_rate': sampling_rate,
        'data_points': len(ecg_array),
        'peak_count': len(peaks)
    }

    # 5. 简单的早搏检测
    premature_count = 0
    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / sampling_rate
        mean_rr = np.mean(rr_intervals)

        # 检测过短的RR间期
        short_intervals = rr_intervals < 0.7 * mean_rr
        premature_count = np.sum(short_intervals)

    # 6. 简单的ST段分析（假设）
    # 这是一个非常简化的版本
    st_segments = []
    for i in range(len(peaks)):
        if i < len(peaks) - 1:
            # J点位置（R波后80ms）
            j_point = peaks[i] + int(0.08 * sampling_rate)
            if j_point < len(ecg_array) - 1:
                st_end = j_point + int(0.06 * sampling_rate)
                if st_end < len(ecg_array):
                    st_value = np.mean(filtered[j_point:st_end])
                    baseline = np.mean(filtered[max(0, j_point-20):j_point])
                    st_deviation = st_value - baseline
                    st_segments.append({
                        'start': j_point,
                        'end': st_end,
                        'deviation': st_deviation,
                        'baseline': baseline,
                        'value': st_value
                    })

    # 7. 综合评估
    assessment = {
        'overall_status': '正常',
        'risk_level': '低',
        'recommendations': []
    }

    # 基于心率评估
    if heart_rate < 50 or heart_rate > 100:
        assessment['overall_status'] = '需要关注'
        if heart_rate < 50:
            assessment['recommendations'].append("心率偏慢")
        else:
            assessment['recommendations'].append("心率偏快")

    # 基于早搏评估
    if premature_count > 5:
        assessment['overall_status'] = '需要关注'
        assessment['risk_level'] = '中'
        assessment['recommendations'].append(f"发现{premature_count}个早搏")

    # ST段评估
    if st_segments:
        max_st_deviation = max(abs(s['deviation']) for s in st_segments)
        if max_st_deviation > 0.1:
            assessment['overall_status'] = '需要关注'
            assessment['risk_level'] = '高'
            assessment['recommendations'].append("ST段异常")

    return {
        'statistics': stats,
        'peaks': peaks.tolist(),
        'premature_count': premature_count,
        'st_segments': st_segments,
        'assessment': assessment,
        'filtered_data': filtered.tolist()
    }

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_data.py <数据文件>")
        print("支持格式: JSON, CSV, 或包含数值的文本文件")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(1)

    print(f"正在分析: {file_path}")

    # 加载数据
    ecg_data, sampling_rate = load_ecg_data(file_path)

    # 分析
    results = analyze_ecg(ecg_data, sampling_rate)

    # 保存结果
    output_file = Path(file_path).stem + '_analysis.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析完成!")
    print(f"心率: {results['statistics']['heart_rate']:.0f} bpm")
    print(f"数据点数: {results['statistics']['data_points']}")
    print(f"持续时间: {results['statistics']['duration']:.1f} 秒")
    print(f"检测到R波: {results['statistics']['peak_count']} 个")
    print(f"早搏数量: {results['premature_count']} 个")
    print(f"整体状态: {results['assessment']['overall_status']}")
    print(f"风险等级: {results['assessment']['risk_level']}")
    print(f"结果文件: {output_file}")

if __name__ == "__main__":
    import os
    import re
    main()