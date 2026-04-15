# researching-recsys-papers

搜推广（搜索、推荐、广告）领域科研论文搜索与分析 Skill for Claude Code

## 核心特性

- **多源并行搜索**: 同时检索 arXiv、Semantic Scholar、Papers with Code、DBLP、Google Scholar
- **全文阅读**: 搜索后自动下载并解析论文 PDF，提取完整内容（强制）
- **模型复现**: 基于论文全文生成可运行代码
- **论文问答**: 可针对检索到的论文提问（原理、结构、计算、代码等）
- **检查点验证**: 关键步骤后暂停确认，防止跑偏
- **异常自恢复**: API 失败自动切换数据源，不中断流程

## 安装

### 方式一：直接复制到 Claude Code skills 目录

```bash
# 克隆仓库
git clone https://github.com/lla0727/researching-recsys.git

# 复制到 Claude Code skills 目录
cp -r researching-recsys ~/.claude/skills/
```

### 方式二：通过 GitHub 导入

在 Claude Code 中使用 `/skill` 命令加载此 Skill

## 依赖

- Python 3.8+
- 标准库（urllib, ssl, json, dataclasses）
- `pypdf` 库（用于提取 PDF 内容）

```bash
pip install pypdf  # 自动安装（脚本会在需要时提示安装）
```

## 快速参考

| 场景 | 输入示例 | 核心动作 |
|------|---------|---------|
| 搜索论文 | "序列建模最新进展" | → 步骤1-7，强制读全文 |
| 论文问答 | "这篇的Attention怎么算" | → 基于已下载全文回答 |
| 获取BibTeX | "给我这篇的引用" | → S2 API / arXiv bibtex端点 |

## 工作流程（7步骤）

```
用户描述问题
     ↓
步骤1: 理解场景（提取领域、问题、已知方案）
     ↓
步骤2: 多源并行检索（5个数据源）
     ↓ 🔍 检查点1：检索结果确认（用户确认后继续）
     ↓
步骤3: 下载并阅读论文全文（强制）
     ↓ 🔍 检查点2：PDF下载确认
     ↓
步骤4: 分析论文（含验证检查点）
     ↓ 🔍 检查点3：分析结果校验
     ↓
步骤5: 输出复现信息（架构/公式/代码）
     ↓
步骤6: 改进建议（含验证：是否回答了用户问题）
     ↓
步骤7: 最终输出确认（三项核对）
```

## 异常处理

| 失败场景 | 处理方式 |
|---------|---------|
| API 超时/429 | 等待 15s 切换其他数据源 |
| 结果为空 | 换关键词或换数据源 |
| curl 报错 | 直接告知用户"无法检索" |
| 连续3次无改善 | 停止该数据源，汇总已有结果 |

## 使用方法

### 基本搜索

```
/researching-recsys-papers
```

然后描述你的研究场景，例如：
- "我想优化推荐系统的序列建模能力"
- "我们在做广告点击率预估，特征交互效果不好"
- "多目标优化在推荐系统中怎么平衡精度和多样性"

### 命令行搜索

```bash
python scripts/search_papers.py "序列推荐" --start-year 2024 --end-year 2026 --max-per-source 10
```

### 下载论文全文

```bash
# 方式1：使用脚本函数
python -c "
from scripts.search_papers import download_and_extract_pdf
text = download_and_extract_pdf('http://arxiv.org/abs/2509.17361v1')
print(text[:2000])
"

# 方式2：直接用 curl
curl -s -L "https://arxiv.org/pdf/2509.17361.pdf" -o /tmp/paper.pdf
```

### Google Scholar 搜索（需要 API Key）

```bash
python scripts/search_papers.py "推荐系统" --serper-key YOUR_SERPER_API_KEY
```

## 文件结构

```
researching-recsys/
├── SKILL.md                              # Skill 定义文件（7步工作流+3个检查点）
├── README.md                             # 本文件
├── scripts/
│   └── search_papers.py                  # 多源论文搜索 + PDF 下载脚本
├── references/
│   ├── paper-reading-guide.md            # 论文阅读指南
│   └── dimension-calculator.md           # 模型维度计算参考
└── demo/
    ├── demo-report.md                    # 示例分析报告
    └── fesail.py                         # FeSAIL 复现代码（增量学习）
```

## 搜索数据源

| 数据源 | API | 说明 |
|--------|-----|------|
| arXiv | export.arxiv.org | 预印本论文 |
| Semantic Scholar | api.semanticscholar.org | 学术元数据+引用 |
| Papers with Code | api.paperswithcode.com | 论文+代码+基准 |
| DBLP | api.dblp.org | 计算机会议/期刊 |
| Google Scholar | Serper.dev | 综合学术搜索（需Key）|

## 输出格式

搜索成功后返回：
1. **场景理解** - 分析你的研究问题
2. **论文列表** - 附论文链接、全文状态标记
3. **论文详情** - 架构、公式、代码（基于全文）
4. **改进方向建议** - 含具体做法/预期收益/实现复杂度
5. **模型复现** - 完整 PyTorch 代码

## 更新日志

### v1.4 (2026-04-15)
- 增加检索结果确认检查点
- 增加PDF下载确认检查点
- 增加分析结果校验检查点
- 增加最终输出确认（步骤7）
- 增加TL;DR快速参考表
- 增加异常处理Fallback表
- 增加改进建议三要素（做法/收益/复杂度）
- 优化参考资源结构

## License

MIT