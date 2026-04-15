# ECG 分析器

ECG波形数据分析工具，支持早搏检测、ST段分析等功能。

## 重要说明

❗ **本工具不直接解析PDF文件**！需要用户提供纯净的ECG数值数据。

## 功能特点

- 📊 **专业分析**: 心率、心律、早搏、ST段等指标
- 🚫 **去噪增强**: 自动信号处理，提高分析准确性
- 📋 **详细报告**: 生成专业ECG分析报告
- ⚠️ **健康预警**: 异常指标自动提醒
- 🔧 **多格式支持**: JSON、CSV、数组格式

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方式1：直接输入数据

直接在对话中提供ECG数据：
```
帮我分析这段ECG数据：[0.1, 0.2, -0.1, 0.3, -0.2, 0.4, ...]
```

### 方式2：上传文件

上传JSON或CSV文件，内容格式如：
```json
{
    "sampling_rate": 125,
    "ecg_data": [0.1, 0.2, -0.1, ...]
}
```

或CSV格式：
```csv
ecg_value
0.1
0.2
-0.1
...
```

### 方式3：命令行使用（需要Python环境）

```bash
# 分析数据文件
python scripts/main.py path/to/data.json
```

## 从PDF中提取数据

由于各品牌手表的PDF格式复杂，推荐使用以下方法提取数据：

1. **手动复制**：如果PDF包含纯文本数据，手动复制数值
2. **在线工具**：使用PDF转CSV工具
3. **专业软件**：如Adobe Acrobat、PDF-XChange Editor等

### 作为Skill使用

1. 上传ECG PDF文件
2. 触发Skill（说"分析ECG"、"查看心电图"等）
3. 获得详细分析报告

## 文件结构

```
ecg-analyzer/
├── SKILL.md                 # Skill定义
├── requirements.txt         # 依赖包列表
├── README.md               # 说明文档
├── scripts/                 # 核心脚本
│   ├── pdf_extractor.py    # PDF数据提取
│   ├── ecg_processor.py     # 信号处理
│   ├── ecg_analyzer.py     # ECG分析算法
│   └── main.py            # 主程序
├── references/             # 参考文档
│   ├── ecg_guidelines.md   # ECG分析标准
│   └── pdf_formats.md     # PDF格式说明
└── assets/                # 资源文件
    └── templates/         # 报告模板
        └── report_template.md
```

## 输出报告

分析报告包含：
- 基础指标（心率、间期等）
- 心律分析
- 早搏检测详情
- ST段分析
- 信号质量评估
- 综合健康建议

## 注意事项

⚠️ **重要提醒**:
- 本工具为辅助分析工具，不能替代专业医疗诊断
- 如有心脏不适，请及时就医
- 分析结果仅供参考，请结合临床症状判断
- 建议在医生指导下使用本工具

## 支持的设备

- Apple Watch Series 4及以上
- 华为手表（心脏健康研究功能）
- 小米手环/手表
- 三星Galaxy Watch
- 其他支持ECG导出的智能设备

## 技术架构

1. **PDF解析**: 使用PyMuPDF提取波形数据
2. **信号处理**: 基于scipy的滤波去噪算法
3. **ECG分析**: 实现标准心电图分析算法
4. **报告生成**: 模板驱动的分析报告

## 开发说明

本Skill采用模块化设计，各组件独立可扩展：
- 新增品牌支持：扩展`pdf_extractor.py`
- 改进算法：更新`ecg_analyzer.py`
- 自定义报告：修改`templates/`下的模板文件

## 许可证

仅供个人学习使用，不得用于商业用途。