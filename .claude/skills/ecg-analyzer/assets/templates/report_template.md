# ECG 分析报告

## 基本信息
- **检测时间**: {{timestamp}}
- **设备品牌**: {{brand}}
- **设备型号**: {{device}}
- **采样率**: {{sampling_rate}} Hz

---

## 📊 基础指标

| 指标 | 数值 | 正常范围 | 状态 |
|------|------|----------|------|
| 心率 | {{heart_rate}} bpm | 60-100 | {{hr_status}} |
| 平均心率 | {{mean_hr}} bpm | 60-100 | {{hr_mean_status}} |
| 心率变异性 | {{hr_std}}% | <10% | {{hr_var_status}} |
| PR间期 | {{pr_interval}} ms | 120-200 | {{pr_status}} |
| QT间期 | {{qt_interval}} ms | <440男/<460女 | {{qt_status}} |
| QTc间期 | {{qtc_interval}} ms | <430男/<450女 | {{qtc_status}} |

---

## ❤️ 心律分析

- **心律类型**: {{rhythm_type}}
- **心律规则性**: {{rhythm_regularity}}
- **不规则搏动数**: {{irregular_beats}} 个
- **RR间期变异性**: {{rr_variability}}

---

## ⚠️ 早搏检测

- **早搏总数**: {{premature_count}} 个
- **早搏类型**: {{premature_type}}
- **早搏率**: {{premature_rate}}%

{% if premature_beats %}
### 早搏详情
| 序号 | 位置 | 类型 |
|------|------|------|
{% for beat in premature_beats %}
| {{loop.index}} | {{beat[0]}} | {{beat[1]}} |
{% endfor %}
{% endif %}

---

## 📈 ST段分析

- **ST段状态**: {{st_status}}
- **平均ST段偏移**: {{st_deviation}} mV
- **最大ST段偏移**: {{max_st_deviation}} mV
- **缺血风险**: {{ischemia_risk}}

{% if st_segments %}
### ST段详情
| 段落 | 起始点 | 终止点 | 偏移(mV) | 基线(mV) | 值(mV) |
|------|--------|--------|----------|----------|--------|
{% for segment in st_segments %}
| {{loop.index}} | {{segment.start}} | {{segment.end}} | {{segment.deviation}} | {{segment.baseline}} | {{segment.value}} |
{% endfor %}
{% endif %}

---

## 🔍 信号质量分析

- **信号质量**: {{signal_quality}}
- **信噪比**: {{signal_quality.snr_db}} dB
- **信号功率**: {{signal_quality.signal_power}}
- **噪声功率**: {{signal_quality.noise_power}}

---

## 🏥 综合评估

### 总体状况
- **健康状态**: {{overall_status}}
- **风险等级**: {{risk_level}}

### 建议
{% for recommendation in recommendations %}
- {{recommendation}}
{% endfor %}

### 重要提醒
⚠️ 本分析结果仅供参考，不能替代专业医疗诊断。如有心脏不适，请及时就医。

---

## 技术说明

本报告由ECG分析器自动生成，分析基于：
- 心电图分析标准
- 信号处理算法
- 医学指南参考

生成时间: {{generate_time}}