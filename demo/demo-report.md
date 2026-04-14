# 搜推广科研分析报告

## 场景理解
用户想要了解序列推荐（Sequence Recommendation）领域的最新研究进展，特别关注深度学习在序列建模中的应用。

## 最新相关论文 (5篇) ⭐2024-2025

### ⭐论文1：《Multi-Aspect Cross-modal Quantization for Generative Recommendation》[2025]
- **来源**：arXiv
- **核心贡献**：提出 MACRec，将多模态信息融入生成式推荐的语义ID学习和生成模型训练
- **论文链接**：http://arxiv.org/abs/2511.15122v2

### ⭐论文2：《OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System》[2025]
- **来源**：arXiv
- **核心贡献**：将 LLM 风格的上下文工程和多步推理引入工业级排序系统，已在 Shopee 部署
- **论文链接**：http://arxiv.org/abs/2509.18091v1

### ⭐论文3：《SeqUDA-Rec: Sequential User Behavior Enhanced Recommendation》[2025]
- **来源**：arXiv
- **核心贡献**：结合全局无监督数据增强和 GAN 增强，在 NDCG@10 提升 6.7%
- **论文链接**：http://arxiv.org/abs/2509.17361v1

### ⭐论文4：《SemSR: Semantics aware robust Session-based Recommendations》[2025]
- **来源**：arXiv
- **核心贡献**：利用 LLM 增强会话推荐，融合语义信息和数据驱动模型
- **论文链接**：http://arxiv.org/abs/2508.20587v1

### ⭐论文5：《M-$LLM^3$REC: A Motivation-Aware User-Item Interaction Framework》[2025]
- **来源**：arXiv
- **核心贡献**：通过 LLM 提取用户动机信号，提升冷启动场景表现
- **论文链接**：http://arxiv.org/abs/2508.15262v1

## 改进方向建议

1. **多模态融合** ⭐来源：论文1
   - 具体做法：将文本/图像等多模态信息量化建模
   - 预期收益：减少语义ID冲突，提高代码本利用率

2. **LLM+推荐系统** ⭐来源：论文2、4、5
   - 具体做法：利用大语言模型提取语义或做上下文增强
   - 预期收益：增强冷启动和稀疏数据场景效果

3. **数据增强+对比学习** ⭐来源：论文3
   - 具体做法：构建全局交互图，用 GAN 生成增强样本
   - 预期收益：NDCG@10 提升 6.7%，提高模型鲁棒性

## 模型复现 (Top 2)

### 复现1：《SemSR》核心架构

```
用户会话 → LLM语义编码 → 数据驱动SR模型 → 融合推荐
     ↓
会话意图识别（粗粒度召回）+ 精细排序
```

### 复现2：《SeqUDA-Rec》核心模块

```python
# 全局用户-物品交互图
GUIG = build_global_graph(all_sequences)

# 图对比学习模块
embeddings = graph_contrastive_learning(GUIG)

# 序列Transformer编码
user_pref = transformer_encoder(behavior_sequence)

# GAN增强
augmented_data = gan_augment(user_pref)
```

---

**搜索耗时**：29.8s（arXiv 成功返回5篇，其他 API 因限流/SSL 失败）
**生成时间**：2026-04-14
