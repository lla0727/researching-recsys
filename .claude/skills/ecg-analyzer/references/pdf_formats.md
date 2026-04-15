# 智能手表ECG PDF格式说明

## Apple Watch ECG PDF

### 文件特征
- **文件名**: 通常包含"ECG"、"Electrocardiogram"等关键词
- **生成方式**: 通过Health App导出
- **页面结构**: 多页，包含图表和文本数据

### 数据提取要点
1. **波形图区域**: PDF中通常包含可视化波形图
   - 位置: 页面中上部
   - 特征: 绿色波形，带有网格背景
   - 尺寸: 约600×300像素

2. **文本数据**: 包含数值信息
   - 心率数值
   - 测量时间戳
   - 心律分类结果
   - 医疗声明文本

3. **元数据**
   ```python
   # 示例元数据字段
   {
       "device": "Apple Watch Series 4+",
       "os": "watchOS 6.0+",
       "app": "ECG App",
       "version": "1.0"
   }
   ```

### 解析策略
- 使用PyMuPDF提取页面内容
- 识别特定区域的数据
- 波形数据可能需要从图像中提取或从文本解析

---

## 华为手表ECG PDF

### 文件特征
- **文件名**: 包含"心电"、"ECG"等
- **应用**: 心脏健康研究或心电分析
- **格式**: 标准PDF，可能包含中英文内容

### 数据结构
1. **报告头部**
   - 患者信息
   - 测量时间
   - 设备型号

2. **ECG波形**
   - 单导联ECG
   - 蓝色或黑色波形
   - 采样率: 125Hz

3. **分析结果**
   - 心率值
   - 心律判断
   - 异常提示

### 特殊注意事项
- 部分华为PDF包含加密内容
- 需要特定权限才能提取数据
- 波形可能与文字重叠

---

## 小米手表ECG PDF

### 文件特征
- **应用**: 心率、心电功能
- **设备**: Mi Band 6、小米手表等
- **格式**: 简化版PDF报告

### 数据内容
1. **基础信息**
   - 测量日期时间
   - 设备信息
   - 用户ID

2. **ECG数据**
   - 简化波形图
   - 数值序列
   - 时间戳

3. **分析结论**
   - 心率范围
   - 是否正常
   - 简单建议

### 解析难点
- 数据量较小
- 可能缺少详细波形
- 需要从图表中重建数据

---

## 三星Galaxy Watch ECG PDF

### 文件特征
- **应用**: Samsung Health Monitor
- **格式**: 专业医疗报告风格

### 数据结构
1. **医疗信息**
   - 患者基本信息
   - 测量目的
   - 临床背景

2. **完整ECG**
   - 12导联模拟（某些型号）
   - 高分辨率波形
   - 时间标记

3. **专业分析**
   - 医生解读（如有）
   - 诊断建议
   - 后续检查建议

---

## 通用PDF解析指南

### 步骤1: 检测PDF类型
```python
def detect_pdf_type(pdf_path):
    # 读取PDF内容
    # 检测关键词
    # 返回品牌类型
```

### 步骤2: 提取波形数据
```python
def extract_waveform(pdf_path, brand):
    if brand == 'apple':
        # Apple特定提取逻辑
    elif brand == 'huawei':
        # 华为特定提取逻辑
    # ...
```

### 步骤3: 数据清洗
```python
def clean_ecg_data(raw_data):
    # 去除噪声
    # 滤波处理
    # 标准化
    return cleaned_data
```

### 常见问题处理

1. **PDF加密**
   - 部分手表PDF可能加密
   - 需要密码或特殊处理

2. **图像PDF**
   - 纯图像，无法直接提取文本
   - 需要OCR或图像处理

3. **格式变化**
   - 不同系统版本格式不同
   - 需要自适应解析

---

## 数据格式标准

### 输出数据格式
```json
{
    "brand": "apple",
    "device": "Apple Watch Series 6",
    "timestamp": "2024-01-01T10:00:00Z",
    "sampling_rate": 128,
    "ecg_data": [0.1, 0.2, -0.1, ...],
    "heart_rate": 72,
    "rhythm": "Sinus Rhythm",
    "metadata": {
        "duration": 10,
        "lead": "I",
        "quality": "Good"
    }
}
```

### 错误处理
```python
class ECGRawPDFError(Exception):
    """PDF解析错误基类"""
    pass

class PDFEncryptedError(ECGRawPDFError):
    """PDF加密错误"""
    pass

class UnsupportedFormatError(ECGRawPDFError):
    """不支持的格式错误"""
    pass
```