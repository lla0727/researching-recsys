# researching-recsys-papers

搜推广（搜索、推荐、广告）领域科研论文搜索与分析 Skill for Claude Code

## 功能特性

- **多源并行搜索**: 同时检索 arXiv、Semantic Scholar、Papers with Code、DBLP、Google Scholar
- **实时在线检索**: 每次搜索调用真实 API，不使用本地/训练数据知识
- **智能去重排序**: 基于来源优先级、年份、引用数自动排序
- **论文问答**: 可针对检索到的论文提问（原理、结构、计算、代码等）
- **模型复现信息**: 提供核心架构、维度变化、参考代码

## 安装

### 方式一：直接复制到 Claude Code skills 目录

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/researching-recsys-papers.git

# 复制到 Claude Code skills 目录
cp -r researching-recsys-papers ~/.claude/skills/
```

### 方式二：通过 GitHub 导入

在 Claude Code 中使用 `/skill` 命令加载此 Skill

## 依赖

- Python 3.8+
- 标准库（urllib, ssl, json, dataclasses）
- 可选：`requests` 库（用于更好的 HTTPS 支持）

```bash
pip install requests  # 可选，但推荐安装
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

### Google Scholar 搜索（需要 API Key）

```bash
python scripts/search_papers.py "推荐系统" --serper-key YOUR_SERPER_API_KEY
```

## 文件结构

```
researching-recsys-papers/
├── SKILL.md                              # Skill 定义文件
├── README.md                             # 本文件
├── scripts/
│   └── search_papers.py                  # 多源论文搜索脚本
└── references/
    ├── paper-reading-guide.md            # 论文阅读指南
    └── dimension-calculator.md          # 模型维度计算参考
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
2. **最新论文列表** - 附论文链接和来源标注
3. **改进方向建议** - 可行动的优化方向
4. **模型复现** - 核心架构和代码

## 上传到 GitHub

### 首次发布

```bash
cd researching-recsys-papers
git init
git add .
git commit -m "Initial commit: researching-recsys-papers skill"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/researching-recsys-papers.git
git push -u origin main
```

### 自动化发布（使用 GitHub Actions）

推送时自动：
1. 运行测试
2. 更新版本号
3. 创建 Release

## License

MIT
