"""
Oppo ECG PDF 数据提取工具
专门用于提取Oppo手表导出的ECG数据
"""

import sys
import os
import re
import json
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """从PDF中提取所有文本"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        print("错误: 请先安装 PyMuPDF")
        print("运行: pip install PyMuPDF")
        sys.exit(1)
    except Exception as e:
        print(f"读取PDF时出错: {e}")
        return ""

def find_ecg_data(text):
    """从文本中寻找ECG数据"""
    # 查找可能的数值模式
    patterns = [
        # 模式1: 数值列表 (如: 0.15, 0.20, -0.10, ...)
        r'([+-]?\d+\.?\d*\s*,\s*[+-]?\d+\.?\d*\s*,\s*[+-]?\d+\.?\d*)',
        # 模式2: 独立数值行
        r'^\s*([+-]?\d+\.?\d*)\s*$',
        # 模式3: 可能的ECG值
        r'\b([+-]?\d+\.?\d*)\b',
    ]

    data = []
    lines = text.split('\n')

    for line in lines:
        # 检查是否有数值列表
        match1 = re.search(patterns[0], line)
        if match1:
            # 提取数值列表
            numbers = re.findall(r'[+-]?\d+\.?\d*', match1.group(1))
            data.extend([float(n) for n in numbers if -10 < float(n) < 10])  # 合理范围
            continue

        # 检查单独的数值
        for num_match in re.finditer(patterns[2], line):
            num = float(num_match.group(1))
            if -10 < num < 10:  # ECG电压的合理范围
                data.append(num)

    return data

def extract_metadata(text):
    """提取元数据"""
    info = {
        'brand': 'oppo',
        'time': '',
        'heart_rate': 0,
        'sampling_rate': 125,  # Oppo常见采样率
    }

    # 查找时间
    time_patterns = [
        r'(\d{4}[-年]\d{1,2}[-月]\d{1,2}日\s*\d{1,2}:\d{2}:\d{2})',
        r'(\d{4}/\d{1,2}/\d{1,2}\s*\d{1,2}:\d{2}:\d{2})',
        r'(\d{1,2}:\d{2}:\d{2})',
    ]

    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match:
            info['time'] = match.group(1)
            break

    # 查找心率
    hr_patterns = [
        r'心率[：:]\s*(\d+)\s*bpm?',
        r'HR[：:]\s*(\d+)',
        r'(\d+)\s*bpm',
    ]

    for pattern in hr_patterns:
        match = re.search(pattern, text)
        if match:
            info['heart_rate'] = int(match.group(1))
            break

    return info

def main():
    if len(sys.argv) < 2:
        print("使用方法: python extract_oppo_ecg.py <PDF文件路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}")
        sys.exit(1)

    print(f"正在分析: {pdf_path}")

    # 1. 提取文本
    print("正在提取PDF文本...")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("警告: PDF中没有找到文本内容")
        print("可能是纯图像PDF，需要使用OCR工具")
        return

    # 2. 提取ECG数据
    print("正在搜索ECG数据...")
    ecg_data = find_ecg_data(text)

    if not ecg_data:
        print("没有找到ECG数据")
        print("请检查PDF是否包含可提取的数值")
        return

    # 3. 提取元数据
    metadata = extract_metadata(text)

    # 4. 创建输出
    output = {
        **metadata,
        'ecg_data': ecg_data,
        'data_points': len(ecg_data),
        'duration': len(ecg_data) / metadata['sampling_rate'],
        'unit': 'mV'
    }

    # 5. 保存结果
    output_file = Path(pdf_path).stem + '_ecg_data.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 成功提取ECG数据!")
    print(f"数据点数: {len(ecg_data)}")
    print(f"持续时间: {output['duration']:.1f} 秒")
    print(f"心率: {metadata['heart_rate']} bpm")
    print(f"输出文件: {output_file}")

    # 显示前10个数据点作为示例
    print(f"\n前10个数据点: {ecg_data[:10]}")

if __name__ == "__main__":
    main()