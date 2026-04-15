# researching-recsys-papers

搜推广（搜索、推荐、广告）领域科研论文搜索与分析 Skill for Claude Code

## 核心特性

- **多源并行搜索**: 同时检索 arXiv、Semantic Scholar、Papers with Code、DBLP、Google Scholar
- **全文阅读**: 搜索后自动下载并解析论文 PDF，提取完整内容
- **模型复现**: 基于论文全文生成可运行代码
- **论文问答**: 可针对检索到的论文提问（原理、结构、计算、代码等）

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

## 工作流程

```
用户描述问题
     ↓
多源并行论文检索（arXiv, Semantic Scholar, Papers with Code, DBLP）
     ↓
下载论文 PDF（自动）
     ↓
提取全文内容（pypdf 解析）
     ↓
基于全文生成：
  - 论文分析报告
  - 核心架构图
  - 关键公式
  - 可运行代码
     ↓
用户可追问论文细节（基于全文内容回答）
```

## 文件结构

```
researching-recsys/
├── SKILL.md                              # Skill 定义文件（包含完整工作流）
├── README.md                             # 本文件
├── scripts/
│   └── search_papers.py                  # 多源论文搜索 + PDF 下载脚本
├── references/
│   ├── paper-reading-guide.md            # 论文阅读指南
│   └── dimension-calculator.md          # 模型维度计算参考
└── demo/
    ├── demo-report.md                    # 示例分析报告
    └── sequda_rec.py                     # SeqUDA-Rec 复现代码
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
4. **改进方向建议** - 可行动的优化方向
5. **模型复现** - 完整 PyTorch 代码

## 上传到 GitHub

```bash
cd researching-recsys
git add .
git commit -m "Update: add full paper reading workflow"
git push
```

## License

MIT
