# 模型维度计算参考

## 常用层维度计算

### Embedding

```
参数量 = vocab_size × embed_dim
输出形状: (batch_size, seq_len, embed_dim)
```

### Linear / Dense

```
参数量 = input_dim × output_dim + output_dim (bias)
输出形状: (batch_size, ..., output_dim)
```

### Attention

```
Q = X × W_Q,  K = X × W_K,  V = X × W_V

Multi-Head Attention:
- d_model: 模型维度
- num_heads: 注意力头数
- d_k = d_model / num_heads

参数量:
- W_Q: d_model × d_k × num_heads
- W_K: d_model × d_k × num_heads
- W_V: d_model × d_k × num_heads
- W_O: d_model × d_model

总计: 4 × d_model²
```

### GRU / LSTM

```
GRU:
- 更新门: input_dim + hidden_dim → hidden_dim
- 重置门: input_dim + hidden_dim → hidden_dim
- 候选: input_dim + hidden_dim → hidden_dim

参数量: 3 × (input_dim + hidden_dim) × hidden_dim

LSTM:
- 遗忘门、输入门、输出门、候选: 4 × (input_dim + hidden_dim) × hidden_dim
```

### CNN

```
二维卷积:
参数量 = kernel_size × in_channels × out_channels + out_channels
输出形状: (batch, out_channels, height, width)
```

## 推荐系统经典模型维度

### DIN (Deep Interest Network)

```
用户行为序列:
- item_id: (M,) → Embedding(M, 64)
- category: (C,) → Embedding(C, 32)
- brand: (B,) → Embedding(B, 32)

Attention Score:
- query: 候选商品 embedding (96,)
- key: 历史商品 embedding (96,)
- 输出: softmax(96 × 96^T) → attention weight
```

### DIEN (Deep Interest Evolution Network)

```
序列层:
- GRU: input=96, hidden=96
- Attention: 96维 query, key, value

输出: 最后一步GRU隐藏状态 (96,)
```

### DeepFM

```
FM部分:
- 一阶: sparse features → Linear
- 二阶: Embedding pairwise interaction

DNN部分:
- 输入: concat(all embeddings)
- 结构: 200 → 400 → 400 → 1
```

### xDeepFM

```
CIN (Compressed Interaction Network):
- 输入: (batch, field_num, embed_dim)
- 第k层: (batch, field_num^k, embed_dim)
- 复杂度: O(m × d × T)
```

## 维度对齐检查清单

- [ ] Embedding维度在整个模型中一致
- [ ] Attention的Q、K、V维度匹配
- [ ] 全连接层输入输出维度正确
- [ ] 多头注意力的d_model能被num_heads整除
- [ ] 序列模型的time_step正确传递

## 常见错误

### 维度不匹配

```python
# 错误
x = self.attention(query=x, key=x, value=x)  # 默认可能维度不对

# 正确
x = self.attention(query=x, key=x, value=x, need_weights=True)
```

### Embedding维度问题

```python
# 错误：vocab_size设置过小
embedding = nn.Embedding(100, 64)  # 如果实际vocab有10000

# 正确：根据实际vocab大小
embedding = nn.Embedding(vocab_size=10000, embedding_dim=64)
```
