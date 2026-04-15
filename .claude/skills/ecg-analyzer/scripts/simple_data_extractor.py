"""
简化的ECG数据提取工具
适用于文字型PDF或提供数据样本的情况
"""

import json
import re
from pathlib import Path

def extract_data_from_text(text):
    """
    从文本中提取ECG数据
    """
    # 查找ECG数据模式
    data = []

    # 模式1: 逗号分隔的数值
    comma_pattern = r'([+-]?\d+\.?\d*(?:\s*,\s*[+-]?\d+\.?\d*)+)'
    matches = re.findall(comma_pattern, text)
    for match in matches:
        numbers = re.findall(r'[+-]?\d+\.?\d*', match)
        data.extend([float(n) for n in numbers])

    # 模式2: 独立数值行
    for line in text.split('\n'):
        numbers = re.findall(r'[+-]?\d+\.?\d+', line)
        data.extend([float(n) for n in numbers])

    return data

def create_sample_data():
    """
    创建模拟ECG数据用于测试
    """
    import math

    # 生成10秒的ECG模拟数据，125Hz采样率
    sampling_rate = 125
    duration = 10  # 秒
    num_points = sampling_rate * duration

    # 模拟ECG波形
    ecg_data = []
    for i in range(num_points):
        t = i / sampling_rate

        # 基础心跳波形（简化版）
        heart_rate = 75  # bpm
        beat_period = 60 / heart_rate

        # 在每个心跳周期内生成波形
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

        # 添加一些噪声
        noise = 0.02 * (math.random() - 0.5) if hasattr(math, 'random') else 0
        value += noise

        ecg_data.append(round(value, 3))

    return ecg_data

def main():
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("1. python simple_data_extractor.py <文本文件>  # 从文本文件提取")
        print("2. python simple_data_extractor.py --sample    # 生成模拟数据")
        sys.exit(1)

    if sys.argv[1] == "--sample":
        # 生成模拟数据
        print("生成模拟ECG数据...")
        ecg_data = create_sample_data()

        output = {
            'brand': 'oppo',
            'sampling_rate': 125,
            'duration': 10,
            'heart_rate': 75,
            'ecg_data': ecg_data,
            'data_points': len(ecg_data),
            'unit': 'mV',
            'note': '这是模拟数据，用于测试分析功能'
        }

        output_file = 'sample_ecg_data.json'

    else:
        # 从文件提取
        file_path = sys.argv[1]

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            sys.exit(1)

        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 提取数据
        ecg_data = extract_data_from_text(text)

        if not ecg_data:
            print("没有找到ECG数据")
            print("\n请确保文件中包含数值数据，例如：")
            print("0.15, 0.20, -0.10, 0.25")
            print("或者")
            print("0.15")
            print("0.20")
            print("-0.10")
            sys.exit(1)

        output = {
            'brand': 'oppo',
            'sampling_rate': 125,
            'ecg_data': ecg_data,
            'data_points': len(ecg_data),
            'duration': len(ecg_data) / 125,
            'heart_rate': 0,  # 需要计算
            'unit': 'mV'
        }

        output_file = Path(file_path).stem + '_ecg.json'

    # 保存JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 提取成功!")
    print(f"数据点数: {len(ecg_data)}")
    print(f"输出文件: {output_file}")

    # 显示样例
    print(f"\n前10个数据点: {ecg_data[:10]}")
    print(f"数据范围: {min(ecg_data):.3f} 到 {max(ecg_data):.3f}")

if __name__ == "__main__":
    import os
    main()