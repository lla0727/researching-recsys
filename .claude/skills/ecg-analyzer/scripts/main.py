"""
ECG 分析主程序
整合PDF解析、信号处理和分析功能
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_extractor import extract_ecg_from_pdf
from ecg_processor import process_ecg_signal
from ecg_analyzer import analyze_ecg


class ECGAnalyzerApp:
    """ECG分析应用主类"""

    def __init__(self):
        self.template_path = Path(__file__).parent.parent / "assets" / "templates" / "report_template.md"

    def analyze_pdf(self, pdf_path: str) -> Dict:
        """分析ECG PDF文件"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'pdf_path': pdf_path,
            'status': 'processing'
        }

        try:
            # 1. 提取ECG数据
            print("正在提取ECG数据...")
            ecg_info = extract_ecg_from_pdf(pdf_path)

            if 'error' in ecg_info:
                results['status'] = 'error'
                results['error'] = ecg_info['error']
                return results

            results.update({
                'brand': ecg_info['brand'],
                'device': f"Unknown {ecg_info['brand']} device",
                'sampling_rate': ecg_info['sampling_rate']
            })

            # 2. 信号处理
            print("正在处理信号...")
            ecg_data = ecg_info['ecg_data']
            if len(ecg_data) == 0:
                results['status'] = 'error'
                results['error'] = 'No ECG data found'
                return results

            processed = process_ecg_signal(ecg_data, ecg_info['sampling_rate'])

            # 3. ECG分析
            print("正在分析ECG...")
            analysis = analyze_ecg(
                processed['processed_signal'],
                processed['r_peaks'],
                processed['sampling_rate']
            )

            # 4. 合并结果
            results.update({
                'status': 'success',
                'heart_rate': analysis['heart_rate'],
                'mean_hr': analysis['mean_hr'],
                'hr_std': analysis['hr_std'],
                'rr_intervals': analysis['rr_intervals'],
                'pr_interval': analysis['pr_interval'],
                'qt_interval': analysis['qt_interval'],
                'qtc_interval': analysis['qtc_interval'],
                'rhythm_type': analysis['rhythm_type'],
                'rhythm_regularity': analysis['rhythm_regularity'],
                'irregular_beats': analysis['irregular_beats'],
                'premature_count': analysis['premature_count'],
                'premature_type': analysis['premature_type'],
                'premature_rate': analysis['premature_rate'],
                'premature_beats': analysis['premature_beats'],
                'st_deviation': analysis['st_deviation'],
                'max_st_deviation': analysis['max_st_deviation'],
                'st_status': analysis['st_status'],
                'ischemia_risk': analysis['ischemia_risk'],
                'overall_status': analysis['overall_status'],
                'risk_level': analysis['risk_level'],
                'recommendations': analysis['recommendations'],
                'signal_quality': processed['signal_quality'],
                'raw_data_length': len(ecg_data),
                'processed_data_length': len(processed['processed_signal'])
            })

            # 5. 生成报告
            print("正在生成报告...")
            report = self._generate_report(results)
            results['report'] = report

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            print(f"分析过程中出错: {e}")

        return results

    def _generate_report(self, results: Dict) -> str:
        """生成分析报告"""
        # 这里可以使用模板引擎，简化起见使用字符串格式化
        report_template = """# ECG 分析报告

## 基本信息
- **检测时间**: {timestamp}
- **设备品牌**: {brand}
- **采样率**: {sampling_rate} Hz

## 📊 基础指标

| 指标 | 数值 | 正常范围 | 状态 |
|------|------|----------|------|
| 心率 | {heart_rate:.0f} bpm | 60-100 | {'正常' if 60 <= heart_rate <= 100 else '异常'} |
| 平均心率 | {mean_hr:.0f} bpm | 60-100 | {'正常' if 60 <= mean_hr <= 100 else '异常'} |
| PR间期 | {pr_interval:.0f} ms | 120-200 | {'正常' if 120 <= pr_interval <= 200 else '异常'} |
| QT间期 | {qt_interval:.0f} ms | <440 | {'正常' if qt_interval < 440 else '异常'} |
| QTc间期 | {qtc_interval:.0f} ms | <430 | {'正常' if qtc_interval < 430 else '异常'} |

## ❤️ 心律分析

- **心律类型**: {rhythm_type}
- **心律规则性**: {rhythm_regularity}
- **不规则搏动数**: {irregular_beats} 个

## ⚠️ 早搏检测

- **早搏总数**: {premature_count} 个
- **早搏类型**: {premature_type}
- **早搏率**: {premature_rate:.1f}%

## 📈 ST段分析

- **ST段状态**: {st_status}
- **平均ST段偏移**: {st_deviation:.3f} mV
- **最大ST段偏移**: {max_st_deviation:.3f} mV
- **缺血风险**: {ischemia_risk}

## 🔍 信号质量分析

- **信号质量**: {signal_quality[quality]}
- **信噪比**: {signal_quality[snr_db]:.1f} dB

## 🏥 综合评估

### 总体状况
- **健康状态**: {overall_status}
- **风险等级**: {risk_level}

### 建议
{% for rec in recommendations %}
- {{rec}}
{% endfor %}

### 重要提醒
⚠️ 本分析结果仅供参考，不能替代专业医疗诊断。如有心脏不适，请及时就医。

---

## 技术说明

本报告由ECG分析器自动生成，分析基于：
- 心电图分析标准
- 信号处理算法
- 医学指南参考

生成时间: {timestamp}
"""

        # 使用简单的字符串替换（实际项目中应使用模板引擎）
        report = report_template.format(**results)

        # 处理多行建议
        if isinstance(results.get('recommendations'), list):
            recommendations_text = "\n".join([f"- {rec}" for rec in results['recommendations']])
            report = report.replace("{% for rec in recommendations %}", "").replace("{% endfor %}", "")
            report = report.replace("- {{rec}}", recommendations_text)

        return report

    def save_results(self, results: Dict, output_dir: str = "output") -> str:
        """保存分析结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(results['pdf_path']).stem
        json_file = os.path.join(output_dir, f"{base_name}_{timestamp}_analysis.json")
        report_file = os.path.join(output_dir, f"{base_name}_{timestamp}_report.md")

        # 保存JSON结果
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存报告
        if 'report' in results:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(results['report'])

        return json_file, report_file


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python main.py <pdf文件路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}")
        sys.exit(1)

    app = ECGAnalyzerApp()
    results = app.analyze_pdf(pdf_path)

    # 保存结果
    json_file, report_file = app.save_results(results)

    print(f"\n分析完成！")
    print(f"详细结果已保存到:")
    print(f"- JSON数据: {json_file}")
    print(f"- 分析报告: {report_file}")

    if results['status'] == 'success':
        print(f"\n分析摘要:")
        print(f"- 心率: {results['heart_rate']:.0f} bpm")
        print(f"- 心律: {results['rhythm_type']}")
        print(f"- 早搏: {results['premature_count']} 个")
        print(f"- ST段: {results['st_status']}")
        print(f"- 整体状态: {results['overall_status']}")
    else:
        print(f"\n分析失败: {results.get('error', '未知错误')}")


if __name__ == "__main__":
    main()