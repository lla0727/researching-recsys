"""
从ECG图像中提取波形数据
使用图像处理技术识别波形
"""

import sys
import json
import numpy as np
from pathlib import Path

def extract_ecg_from_image(image_path):
    """从图像中提取ECG波形数据"""
    try:
        from PIL import Image
        from PIL import ImageDraw
        import matplotlib.pyplot as plt
    except ImportError:
        print("正在安装图像处理库...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        from PIL import Image
        from PIL import ImageDraw

    # 读取图像
    img = Image.open(image_path)

    # 转换为灰度
    if img.mode != 'L':
        img = img.convert('L')

    # 转换为numpy数组
    img_array = np.array(img)

    print(f"图像尺寸: {img_array.shape}")

    # 寻找波形区域
    # 简化算法：寻找水平线条
    height, width = img_array.shape

    # 从每行提取一个数据点（假设波形是水平的）
    ecg_values = []

    # 从中间开始扫描
    center_y = height // 2
    scan_height = min(height, 200)  # 只扫描中间区域

    # 从右到左扫描（ECG通常从右向左或从左向右）
    for x in range(0, width, 2):  # 每2个像素取样一次
        # 在扫描区域内找到最亮或最暗的点
        roi = img_array[center_y-scan_height//2:center_y+scan_height//2, x:x+2]

        # 寻找边缘
        if len(roi.shape) == 2 and roi.shape[1] > 0:
            # 使用梯度找边缘
            if len(roi.shape) == 2 and roi.shape[0] > 1:
                # 找到垂直方向的边缘
                vertical_gradient = np.abs(np.diff(roi, axis=0))
                if len(vertical_gradient) > 0:
                    # 取最强边缘位置
                    max_grad_idx = np.argmax(vertical_gradient)
                    y_pos = center_y - scan_height//2 + max_grad_idx
                    # 归一化到0-1范围
                    normalized_value = 1 - (y_pos / height)
                    ecg_values.append(normalized_value)

    if not ecg_values:
        print("未能从图像中提取到波形数据")
        return []

    # 反转数据（根据波形方向）
    # ECG通常向上为正
    ecg_values = np.array(ecg_values)

    # 去噪
    from scipy.signal import medfilt
    ecg_values = medfilt(ecg_values, kernel_size=3)

    # 缩放到合理范围（mV）
    ecg_values = ecg_values * 2 - 1  # 缩放到-1到1mV范围

    return ecg_values.tolist()

def main():
    if len(sys.argv) < 2:
        print("使用方法: python extract_from_image.py <图像文件>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"文件不存在: {image_path}")
        sys.exit(1)

    print(f"正在处理图像: {image_path}")

    # 提取ECG数据
    ecg_data = extract_ecg_from_image(image_path)

    if not ecg_data:
        print("提取失败")
        sys.exit(1)

    # 保存结果
    output_data = {
        'brand': 'oppo',
        'sampling_rate': 125,  # Oppo常见采样率
        'duration': len(ecg_data) / 125,
        'ecg_data': ecg_data,
        'data_points': len(ecg_data),
        'unit': 'mV',
        'source': 'image_extraction'
    }

    output_file = Path(image_path).stem + '_from_image.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n提取成功!")
    print(f"数据点数: {len(ecg_data)}")
    print(f"持续时间: {len(ecg_data)/125:.1f} 秒")
    print(f"数据范围: {min(ecg_data):.3f} 到 {max(ecg_data):.3f}")
    print(f"输出文件: {output_file}")

    # 显示部分数据
    print(f"\n前20个数据点: {ecg_data[:20]}")

if __name__ == "__main__":
    main()