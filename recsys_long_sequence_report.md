# 搜推广科研分析报告：长序列推荐系统序列建模优化

> 生成时间：2026-04-14
> 检索数据源：arXiv, Semantic Scholar, Papers with Code, dblp
> 检索关键词：long sequence recommendation, lifelong user behavior, sequential recommendation transformer

---

## 场景理解

**核心问题**：用户历史行为序列很长 → 标准 Transformer O(N²d) 复杂度导致延迟爆炸、内存不足，工业部署被迫截断序列，丢失长期兴趣。

**技术挑战三角**：
- 计算效率：Attention 复杂度随序列长度平方增长
- 建模质量：长序列中有效兴趣信号稀疏，噪声多
- 系统延迟：I/O 读取大量历史行为本身就慢

---

## 最新相关论文 (2024-2026) ⭐

### ⭐ 论文1：SOLAR: SVD-Optimized Lifelong Attention for Recommendation [2026]

- **来源**：arXiv | 快手 Kuaishou 团队
- **论文链接**：http://arxiv.org/abs/2603.02561v1
- **核心贡献**：发现推荐系统的 user behavior sequence 矩阵天然是低秩的，利用 SVD 将 Attention 复杂度从 O(N²d) 降至 O(Ndr)，**理论上对低秩矩阵无损**，保留 softmax 机制（不像 linear attention 丢掉 softmax）
- **工业效果**：支持**万级别**行为序列 + 数千候选集，无需提前 filtering；快手在线推荐 **+0.68% Video Views**

| 组件 | 传统 Attention | SVD-Attention |
|------|--------------|---------------|
| 复杂度 | O(N²d) | O(Ndr)，r≪N |
| softmax | ✅ | ✅ (保留) |
| 低秩假设 | ✗ | ✅ (推荐天然低秩) |

---

### ⭐ 论文2：LASER: Target-Aware Segmented Attention for Long Sequence Modeling [2026]

- **来源**：arXiv | 小红书 (RedNote) 团队
- **论文链接**：http://arxiv.org/abs/2602.11562v1
- **核心贡献**：双层优化 — 系统层 (SeqVault) + 算法层 (STA)
  - **SeqVault**：DRAM-SSD 混合索引，检索延迟降低 50%，CPU 使用率降低 75%
  - **STA (Segmented Target Attention)**：sigmoid gating 过滤噪声 item，然后 GSTA 捕捉跨 segment 依赖
- **工业效果**：服务 **1亿日活用户**，+2.36% ADVV，+2.08% revenue

```
长行为序列 → SeqVault(DRAM+SSD混合读取) → 分段(Segment)
→ Sigmoid Gate(过滤噪声) → GSTA(跨段聚合) → 预测层
```

---

### ⭐ 论文3：HiSAC: Hierarchical Sparse Activation Compression [2026]

- **来源**：arXiv | 淘宝 Taobao 团队
- **论文链接**：http://arxiv.org/abs/2602.21009v1
- **核心贡献**：不直接跑全序列，将历史行为压缩成"兴趣中心"
  - 多层级语义 ID (Multi-level Semantic IDs)
  - 层次化投票机制稀疏激活个性化 Interest-Agent
  - Soft-Routing Attention：按语义相似度加权聚合，保留长尾行为
- **工业效果**："猜你喜欢"首页 **+1.65% CTR**，显著压缩序列存储

---

### ⭐ 论文4：FuXi-Linear: Linear Attention for Long-term Sequential Recommendation [2026]

- **来源**：arXiv | USTC
- **论文链接**：http://arxiv.org/abs/2602.23671v1
- **代码**：https://github.com/USTC-StarTeam/fuxi-linear ✅ (开源)
- **核心贡献**：Linear Attention + 时序信号分离
  - **Temporal Retention Channel**：独立计算周期性时序权重，避免时序与语义信号互干扰
  - **Linear Positional Channel**：可学习 kernel 注入位置信息
  - Power-law 扩展律：千级别序列上验证
- **工业效果**：prefill 加速 **10×**，decode 加速 **21×**，效果优于 SOTA

---

### ⭐ 论文5：CollectiveKV: Cross-User KV Sharing for Efficient Sequential Recommendation [2026]

- **来源**：arXiv
- **论文链接**：http://arxiv.org/abs/2601.19178v2
- **核心贡献**：用 SVD 分析 KV Cache，发现不同用户的 KV 有大量共享信息
  - 全局共享 KV Pool（高维，所有用户共享）
  - 用户特有 KV（低维，per-user）
  - 推理时拼接，KV Cache 压缩到原始大小的 **0.8%**
- **适用场景**：用户量大、KV 存储成本高的工业推荐系统

---

### ⭐ 论文6：GEMs: Breaking Long-Sequence Barrier with Multi-Stream Decoder [2026]

- **来源**：arXiv
- **论文链接**：http://arxiv.org/abs/2602.13631v1
- **核心贡献**：生成式推荐 + 多流解码器，突破长序列瓶颈
  - 多流 Decoder 并行处理不同兴趣流
  - 专为生成式推荐框架设计

---

### ⭐ 论文7：HoloMambaRec: Mamba-based Efficient Sequential Recommendation [2026]

- **来源**：arXiv
- **论文链接**：http://arxiv.org/abs/2601.08360v2
- **核心贡献**：将 Mamba (SSM 状态空间模型) 引入序列推荐
  - Holographic Reduced Representation 编码 item 属性
  - Selective State Space Encoder：线性时间推理，恒定推理复杂度
  - 超越 SASRec，内存复杂度更低

---

## 改进方向建议

### 方向1：SVD 低秩近似 Attention（来源：SOLAR）
- **具体做法**：对 user behavior 矩阵做 SVD 低秩分解，将 O(N²d) → O(Ndr)，保留 softmax
- **预期收益**：直接支持万级别序列，无需截断；推荐效果损失几乎为 0
- **实现复杂度**：中等

### 方向2：分层兴趣压缩 + 软路由（来源：HiSAC）
- **具体做法**：将行为聚类为 K 个 Interest-Agent，再用 Soft-Routing Attention 按相似度加权聚合
- **预期收益**：序列存储显著压缩，保留长尾行为，CTR +1-2%
- **实现复杂度**：中等

### 方向3：Sigmoid Gate 分段注意力（来源：LASER）
- **具体做法**：对长序列分段（如每 50 个 item 一段），每段内用 sigmoid gate 过滤噪声，再用轻量 GSTA 跨段聚合
- **预期收益**：计算复杂度从 O(N²) → O(N·K)
- **实现复杂度**：低-中

### 方向4：Linear Attention + 时序分离（来源：FuXi-Linear）
- **具体做法**：时序信号和语义信号走独立的 Attention Channel；有开源代码可参考
- **预期收益**：推理加速 10-21×
- **实现复杂度**：中等（有开源代码，可直接复用）

### 方向5：KV Cache 跨用户共享（来源：CollectiveKV）
- **具体做法**：离线用 SVD 提取用户公共 KV，在线推理时 per-user 只存低维差异部分
- **预期收益**：KV 存储压缩到 0.8%
- **实现复杂度**：高

---

## 模型复现（Top 2）

### 复现1：LASER - Segmented Target Attention (STA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentedTargetAttention(nn.Module):
    def __init__(self, d_model, seg_len=50, n_heads=4):
        super().__init__()
        self.seg_len = seg_len
        self.d_model = d_model
        self.gate_proj = nn.Linear(d_model, 1)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, seq_emb, target_emb):
        """
        seq_emb:    [B, N, D] 长行为序列
        target_emb: [B, D]    候选 item
        """
        B, N, D = seq_emb.shape
        s = self.seg_len

        # 1. 分段（补齐）
        pad_len = (s - N % s) % s
        if pad_len > 0:
            seq_emb = F.pad(seq_emb, (0, 0, 0, pad_len))
        K_segs = seq_emb.shape[1] // s
        segs = seq_emb.view(B, K_segs, s, D)  # [B, K, s, D]

        # 2. 段内 Sigmoid Gate
        target_q = target_emb.unsqueeze(1).unsqueeze(1)   # [B, 1, 1, D]
        gate_input = segs * target_q
        gate_score = torch.sigmoid(self.gate_proj(gate_input))  # [B, K, s, 1]
        filtered = (segs * gate_score).sum(dim=2)              # [B, K, D]

        # 3. GSTA - 跨段 Target Attention
        q = self.q_proj(target_emb).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(filtered)                  # [B, K, D]
        v = self.v_proj(filtered)                  # [B, K, D]

        def to_heads(x):
            B, T, D = x.shape
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        scale = self.head_dim ** -0.5
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).squeeze(2).view(B, D)
        return out
```

---

### 复现2：SOLAR - SVD-Attention 核心

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDAttention(nn.Module):
    """
    SVD-Optimized Attention for low-rank user behavior matrices.
    复杂度: O(Ndr), r << N，保留 softmax
    """
    def __init__(self, d_model, rank_ratio=0.1, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rank_ratio = rank_ratio
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key_value, mask=None):
        """
        query:     [B, Tq, D]  候选 item 或短序列
        key_value: [B, N, D]   长用户行为序列
        """
        B, N, D = key_value.shape
        Tq = query.shape[1]
        r = max(1, int(N * self.rank_ratio))

        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        def reshape_heads(x):
            B, T, D = x.shape
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = reshape_heads(Q), reshape_heads(K), reshape_heads(V)

        K_flat = K.reshape(B * self.n_heads, N, self.head_dim)
        Q_flat = Q.reshape(B * self.n_heads, Tq, self.head_dim)

        try:
            U, S, Vh = torch.linalg.svd(K_flat, full_matrices=False)
            U_r  = U[:, :, :r]     # [B*H, N, r]
            S_r  = S[:, :r]        # [B*H, r]
            Vh_r = Vh[:, :r, :]    # [B*H, r, d]

            # Q·Kᵀ ≈ (Q·Vh_rᵀ)·diag(S_r)·U_rᵀ  — O(Ndr)
            QVhS = (Q_flat @ Vh_r.transpose(-2, -1)) * S_r.unsqueeze(1)
            attn_scores = QVhS @ U_r.transpose(-2, -1)  # [B*H, Tq, N]
        except RuntimeError:
            attn_scores = Q_flat @ K_flat.transpose(-2, -1)

        scale = self.head_dim ** -0.5
        attn_scores = attn_scores * scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        V_flat = V.reshape(B * self.n_heads, N, self.head_dim)
        out = (attn_weights @ V_flat).view(B, self.n_heads, Tq, self.head_dim)
        out = out.transpose(1, 2).reshape(B, Tq, D)
        return self.out_proj(out)
```

---

## 方案对比总结

| 方案 | 核心思路 | 工业效果 | 实现难度 | 开源 |
|------|---------|---------|---------|------|
| **SOLAR** | SVD 低秩 Attention | 快手 +0.68% VV | 中 | ❌ |
| **LASER** | 分段+Sigmoid Gate | 小红书 +2.36% ADVV | 低-中 | ❌ |
| **HiSAC** | 兴趣中心+软路由 | 淘宝 +1.65% CTR | 中 | ❌ |
| **FuXi-Linear** | 时序分离 Linear Attention | 推理加速 10-21× | 中 | ✅ |
| **CollectiveKV** | 跨用户 KV 共享 | KV 压缩至 0.8% | 高 | ❌ |

**推荐落地路径**：
1. **起步快**：FuXi-Linear（有开源代码）验证 linear attention 效果
2. **效果好**：序列 > 500 时，用 LASER 分段 Sigmoid Gate，实现最简单
3. **追求极致**：SOLAR SVD-Attention，适合序列达万级别的场景
