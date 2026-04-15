"""
高级ECG图像提取
使用颜色分离和边缘检测来提取波形
"""

import sys
import json
import numpy as np
from pathlib import Path

def extract_ecg_advanced(image_path):
    """高级ECG波形提取"""
    try:
        from PIL import Image
        import cv2
    except ImportError:
        print("安装OpenCV...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        from PIL import Image
        import cv2

    # 读取图像
    img = Image.open(image_path)

    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 转换为灰度
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    print(f"图像尺寸: {gray.shape}")

    # 颜色阈值分离 - 尝试提取波形线
    # Oppo ECG通常是红色或蓝色波形
    # 先尝试红色通道
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # 红色范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # 蓝色范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 合并颜色掩码
    color_mask = cv2.bitwise_or(mask_red, mask_blue)

    # 去噪
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.erode(color_mask, kernel, iterations=1)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # 边缘检测
    edges = cv2.Canny(color_mask, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"找到 {len(contours)} 个轮廓")

    # 如果没有找到轮廓，使用边缘检测
    if len(contours) < 10:
        print("使用边缘检测方法...")
        # Sobel边缘检测
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)

        # 二值化
        _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找数据点
        height, width = binary.shape

        # 从左到右扫描
        ecg_values = []
        center_y = height // 2

        for x in range(0, width, 2):
            # 在中心垂直线上找最亮的点
            roi = binary[center_y-100:center_y+100, x:x+2]

            if roi.shape[0] > 0 and roi.shape[1] > 0:
                # 找第一个非零像素
                nonzero_y = np.where(roi > 0)[0]
                if len(nonzero_y) > 0:
                    y_pos = center_y - 100 + nonzero_y[0]
                    normalized_value = 1 - (y_pos / height)
                    ecg_values.append(normalized_value)

        if ecg_values:
            ecg_values = np.array(ecg_values) * 2 - 1  # 缩放到-1到1mV
            return ecg_values.tolist()

        return []

    # 使用轮廓方法
    # 找到最大的轮廓（应该是波形）
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取边界矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    print(f"波形区域: x={x}, y={y}, w={w}, h={h}")

    # 从轮廓中提取波形
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 在波形区域内扫描
    ecg_values = []

    for col in range(x, x + w, 2):
        # 在垂直线上找边缘
        roi = mask[y:y+h, col:col+2]
        if roi.shape[0] > 1:
            # 找上边缘
            top_edge = 0
            for row in range(roi.shape[0]):
                if roi[row, 0] > 0:
                    top_edge = row
                    break

            # 找下边缘
            bottom_edge = roi.shape[0] - 1
            for row in range(roi.shape[0]-1, -1, -1):
                if roi[row, 0] > 0:
                    bottom_edge = row
                    break

            # 使用中点
            mid_y = (top_edge + bottom_edge) // 2
            normalized_value = 1 - ((y + mid_y) / gray.shape[0])
            ecg_values.append(normalized_value)

    if not ecg_values:
        print("未能提取波形数据")
        return []

    ecg_values = np.array(ecg_values) * 2 - 1  # 缩放到-1到1mV

    return ecg_values.tolist()

def main():
    if len(sys.argv) < 2:
        print("使用方法: python extract_ecg_advanced.py <图像文件>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"文件不存在: {image_path}")
        sys.exit(1)

    print(f"正在处理图像: {image_path}")

    # 提取ECG数据
    ecg_data = extract_ecg_advanced(image_path)

    if not ecg_data:
        print("提取失败")
        sys.exit(1)

    # 保存结果
    output_data = {
        'brand': 'oppo',
        'sampling_rate': 125,
        'duration': len(ecg_data) / 125,
        'ecg_data': ecg_data,
        'data_points': len(ecg_data),
        'unit': 'mV',
        'source': 'advanced_image_extraction'
    }

    output_file = Path(image_path).stem + '_advanced.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n提取成功!")
    print(f"数据点数: {len(ecg_data)}")
    print(f"持续时间: {len(ecg_data)/125:.1f} 秒")
    print(f"数据范围: {min(ecg_data):.3f} 到 {max(ecg_data):.3f}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    main()