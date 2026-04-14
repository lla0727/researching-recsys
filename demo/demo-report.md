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

## 模型复现 (Top 1)

### 复现：《SeqUDA-Rec》⭐基于论文全文

论文链接：http://arxiv.org/abs/2509.17361v1

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      SeqUDA-Rec Framework                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │  User Item   │                                           │
│  │  Sequences   │                                           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────┐           │
│  │  Module 1: GAN-based Data Augmentation       │           │
│  │  - Generator: LSTM → softmax(item_count)     │           │
│  │  - Discriminator: GRU → Linear(2)           │           │
│  └──────┬───────────────────────────────────────┘           │
│         │  augmented sequences                               │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────┐           │
│  │  Module 2: Global User-Item Graph (GUIG)     │           │
│  │  - User/Item Embeddings                      │           │
│  │  - Graph Contrastive Learning               │           │
│  │    L_CCL = -log exp(sim(h_u,h_u')/T)        │           │
│  │              / Σ exp(sim(h_u,h_v)/T)        │           │
│  └──────┬───────────────────────────────────────┘           │
│         │  robust embeddings                                 │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────┐           │
│  │  Module 3: Transformer + Target Attention    │           │
│  │  - Positional Encoding + Multi-head SA      │           │
│  │  - Target Attention → MLP → CTR/CVR         │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 关键公式

**Graph Contrastive Learning Loss:**
```
L_CCL = - log exp(sim(h_u, h_u')/T) / Σ_v exp(sim(h_u, h_v)/T)
```
- `h_u`: anchor user embedding
- `h_u'`: positive sample (same user's item embedding)
- `h_v`: negative samples (different users)
- `T`: temperature parameter

### 核心代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Generator(nn.Module):
    """Generator: learns distribution of real user-interaction sequences"""
    def __init__(self, item_count, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.item_embed = nn.Embedding(item_count, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, item_count)

    def forward(self, sequence, lengths):
        x = self.item_embed(sequence)
        x, _ = self.lstm(x)
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)


class Discriminator(nn.Module):
    """Discriminator: judges whether sequence is real or generated"""
    def __init__(self, item_count, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.item_embed = nn.Embedding(item_count, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, sequence):
        x = self.item_embed(sequence)
        _, hidden = self.gru(x)
        return self.classifier(hidden.squeeze(0))


class GraphContrastiveLearning(nn.Module):
    """Graph Contrastive Learning with temperature"""
    def __init__(self, embed_dim=64, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def contrastive_loss(self, h_u, h_u_pos, h_v_neg):
        h_u = F.normalize(h_u, dim=-1)
        h_u_pos = F.normalize(h_u_pos, dim=-1)
        h_v_neg = F.normalize(h_v_neg, dim=-1)

        pos_sim = F.cosine_similarity(h_u, h_u_pos, dim=-1) / self.temperature
        neg_sim = torch.bmm(h_u.unsqueeze(1), h_v_neg.transpose(1, 2)).squeeze(1) / self.temperature

        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim).sum(dim=1)
        return -torch.log(exp_pos / (exp_pos + exp_neg)).mean()


class TransformerSequentialEncoder(nn.Module):
    """Transformer Encoder with Positional Encoding"""
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        return self.transformer(x, src_key_padding_mask=mask)


class TargetAttention(nn.Module):
    """Target-Attention: focuses on historical behaviors relevant to candidate"""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, history, target):
        Q = self.query_proj(target).unsqueeze(1)
        K = self.key_proj(history)
        V = self.value_proj(history)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(history.size(-1))
        attn = torch.bmm(F.softmax(scores, dim=-1), V).squeeze(1)
        return attn


class SeqUDA_Rec(nn.Module):
    """Full SeqUDA-Rec Model"""
    def __init__(self, user_count, item_count, embed_dim=64):
        super().__init__()
        self.guig = GlobalUserItemGraph(user_count, item_count, embed_dim)
        self.gcl = GraphContrastiveLearning(embed_dim)
        self.transformer = TransformerSequentialEncoder(embed_dim)
        self.target_attention = TargetAttention(embed_dim)
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, user_ids, item_seq, lengths, target_item):
        user_emb, item_emb = self.guig(user_ids, item_emb)
        seq_output = self.transformer(item_emb)
        target_emb = self.guig.item_embed(target_item)
        attended = self.target_attention(seq_output, target_emb)
        return self.output(attended + user_emb).squeeze(-1)
```

### 实验结果

| Dataset | Metric | BERTRec | GCL4SR | SASRec | SeqUDA-Rec |
|---------|--------|---------|--------|--------|------------|
| Amazon  | HR@10  | 0.638   | 0.655  | 0.621  | **0.709**  |
| Amazon  | NDCG@10| 0.417   | 0.428  | 0.403  | **0.456**  |
| TikTok  | HR@10  | 0.532   | 0.551  | 0.549  | **0.613**  |
| TikTok  | NDCG@10| 0.356   | 0.364  | 0.361  | **0.401**  |

### 完整代码文件

见 `demo/sequda_rec.py`

---

**搜索耗时**：29.8s（arXiv 成功返回5篇，其他 API 因限流/SSL 失败）
**论文原文获取**：通过 curl 下载 arXiv PDF 并解析
**生成时间**：2026-04-14
