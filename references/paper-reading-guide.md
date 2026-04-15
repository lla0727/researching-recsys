# 论文阅读指南

## 快速定位论文价值

### 必读部分（按顺序）

1. **标题 + 摘要**：3分钟内判断是否相关
2. **Introduction 最后一段**：作者自述贡献
3. **Methodology 核心章节**：模型架构
4. **Experiment 表格**：效果提升多少
5. **Conclusion**：限制和未来工作

### 关键问题

- 这篇论文解决了什么问题？
- 比之前的state-of-the-art改进了什么？
- 核心创新点是什么？
- 局限性在哪里？

## 从论文提取复现信息

### 模型架构提取

1. 找到模型结构图（figure）
2. 追踪数据流：输入 → 各层 → 输出
3. 记录每层的：
   - 层类型（Linear, Conv, Attention, GRU等）
   - 输入/输出维度
   - 激活函数
   - 正则化方式

### 维度计算

```
参数量 = 输入维度 × 输出维度 + 偏置

示例：
- Linear(512, 256): 512 × 256 + 256 = 131,328
- Embedding(10000, 64): 10000 × 64 = 640,000
- Multi-Head Attention(d_model=512, heads=8):
  - Q, K, V 投影: 3 × (512 × 512) = 786,432
  - 输出投影: 512 × 512 = 262,144
```

### 代码结构模板

```python
class ModelName(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 核心模块
        self.core_module = CoreLayer(
            input_dim=embed_dim,
            output_dim=hidden_dim,
            heads=num_heads
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.core_module(x)
        return self.output_layer(x)
```

## 顶会论文来源

| 会议 | 领域 | 论文难度 |
|------|------|----------|
| KDD | 数据挖掘 | 高 |
| SIGIR | 信息检索 | 中高 |
| RecSys | 推荐系统 | 中 |
| CIKM | 信息与知识管理 | 中高 |
| AAAI | AI应用 | 中 |
| NeurIPS | 机器学习 | 高 |
| WWW | Web研究 | 中高 |

## 常用论文搜索

### arXiv

- cs.IR (Information Retrieval)
- cs.LG (Machine Learning)
- cs.AI (Artificial Intelligence)

### Semantic Scholar

免费API，支持按引用数排序

### Google Scholar

按相关性/引用数/时间排序
