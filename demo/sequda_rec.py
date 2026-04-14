"""
SeqUDA-Rec: Sequential User Behavior Enhanced Recommendation via Global Unsupervised Data Augmentation
Based on paper: http://arxiv.org/abs/2509.17361v1

Architecture:
1. GAN-based Data Augmentation (Generator + Discriminator)
2. Global User-Item Interaction Graph (GUIG) with Graph Contrastive Learning
3. Transformer-based Sequential Encoder
"""

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
        """
        Args:
            sequence: (batch, seq_len) item indices
            lengths: actual sequence lengths
        Returns:
            generated_sequences: (batch, seq_len, item_count) probabilities
        """
        x = self.item_embed(sequence)  # (batch, seq_len, embed_dim)
        x, _ = self.lstm(x)
        logits = self.output_layer(x)  # (batch, seq_len, item_count)
        return F.softmax(logits, dim=-1)


class Discriminator(nn.Module):
    """Discriminator: judges whether sequence is real or generated"""
    def __init__(self, item_count, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.item_embed = nn.Embedding(item_count, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)  # real=1, fake=0

    def forward(self, sequence):
        """
        Args:
            sequence: (batch, seq_len) item indices
        Returns:
            logits: (batch, 2) real/fake classification
        """
        x = self.item_embed(sequence)
        _, hidden = self.gru(x)
        logits = self.classifier(hidden.squeeze(0))
        return logits


class GlobalUserItemGraph(nn.Module):
    """Global User-Item Interaction Graph (GUIG)"""
    def __init__(self, user_count, item_count, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(user_count, embed_dim)
        self.item_embed = nn.Embedding(item_count, embed_dim)

    def forward(self, user_ids, item_ids):
        """
        Args:
            user_ids: (batch,)
            item_ids: (batch, seq_len)
        Returns:
            user_emb: (batch, embed_dim)
            item_emb: (batch, seq_len, embed_dim)
        """
        user_emb = self.user_embed(user_ids)
        item_emb = self.item_embed(item_ids)
        return user_emb, item_emb


class GraphContrastiveLearning(nn.Module):
    """Graph Contrastive Learning Module with temperature"""
    def __init__(self, embed_dim=64, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Linear(embed_dim, embed_dim)

    def sim(self, h1, h2):
        """Cosine similarity"""
        return F.cosine_similarity(h1, h2, dim=-1)

    def contrastive_loss(self, h_u, h_u_pos, h_v_neg):
        """
        Args:
            h_u: anchor user embedding (batch, embed_dim)
            h_u_pos: positive sample (batch, embed_dim)
            h_v_neg: negative samples (batch, num_neg, embed_dim)
        Returns:
            loss: scalar
        """
        h_u = F.normalize(h_u, dim=-1)
        h_u_pos = F.normalize(h_u_pos, dim=-1)
        h_v_neg = F.normalize(h_v_neg, dim=-1)

        pos_sim = self.sim(h_u, h_u_pos) / self.temperature
        neg_sim = torch.bmm(h_u.unsqueeze(1), h_v_neg.transpose(1, 2)).squeeze(1) / self.temperature

        # L_CCL = - log exp(sim(h_u, h_u')/T) / sum(exp(sim(h_u, h_v)/T))
        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim).sum(dim=1)
        loss = -torch.log(exp_pos / (exp_pos + exp_neg)).mean()

        return loss


class TransformerSequentialEncoder(nn.Module):
    """Transformer-based Sequential Encoder with Multi-head Self-Attention and Target-Attention"""
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: padding mask if needed
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        x = self.positional_encoding(x)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        return output


class TargetAttention(nn.Module):
    """Target-Attention Mechanism: focuses on historical behaviors most relevant to candidate ad"""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, history, target):
        """
        Args:
            history: (batch, seq_len, embed_dim) user behavior sequence
            target: (batch, embed_dim) candidate item
        Returns:
            attended: (batch, embed_dim)
        """
        Q = self.query_proj(target).unsqueeze(1)  # (batch, 1, embed_dim)
        K = self.key_proj(history)  # (batch, seq_len, embed_dim)
        V = self.value_proj(history)  # (batch, seq_len, embed_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(history.size(-1))  # (batch, 1, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.bmm(attn_weights, V).squeeze(1)  # (batch, embed_dim)

        return attended


class SeqUDA_Rec(nn.Module):
    """
    Full SeqUDA-Rec Architecture:
    1. GAN-based Data Augmentation
    2. Global Graph Contrastive Learning
    3. Transformer Sequential Encoder + Target Attention
    """
    def __init__(self, user_count, item_count, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim

        # Module 1: GAN
        self.generator = Generator(item_count, embed_dim, embed_dim * 2)
        self.discriminator = Discriminator(item_count, embed_dim, embed_dim * 2)

        # Module 2: Global Graph
        self.guig = GlobalUserItemGraph(user_count, item_count, embed_dim)
        self.gcl = GraphContrastiveLearning(embed_dim)

        # Module 3: Transformer + Target Attention
        self.transformer = TransformerSequentialEncoder(embed_dim, num_heads, num_layers)
        self.target_attention = TargetAttention(embed_dim)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, user_ids, item_sequence, lengths, target_item):
        """
        Args:
            user_ids: (batch,)
            item_sequence: (batch, seq_len)
            lengths: (batch,) actual lengths
            target_item: (batch,) candidate item
        Returns:
            logits: (batch,) CTR/CVR prediction
        """
        # Get embeddings
        user_emb, item_emb = self.guig(user_ids, item_sequence)

        # Pass through Transformer
        padding_mask = torch.zeros_like(item_sequence, dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        seq_output = self.transformer(item_emb, mask=padding_mask)

        # Target attention
        target_emb = self.guig.item_embed(target_item)
        attended = self.target_attention(seq_output, target_emb)

        # Combine with user and target embeddings
        combined = attended + user_emb + target_emb

        # Output prediction
        logits = self.output_layer(combined).squeeze(-1)
        return logits


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============== Training Step Example ==============

def train_step_gan(batch, model, optimizer_G, optimizer_D):
    """One training step for GAN module"""
    user_ids, item_seq, lengths, target_item, labels = batch

    # === Train Discriminator ===
    optimizer_D.zero_grad()
    real_seq = item_seq
    fake_seq = torch.argmax(model.generator(item_seq, lengths), dim=-1)
    disc_real = model.discriminator(real_seq)
    disc_fake = model.discriminator(fake_seq.detach())

    loss_D = F.cross_entropy(disc_real, torch.ones(len(labels), dtype=torch.long)) + \
             F.cross_entropy(disc_fake, torch.zeros(len(labels), dtype=torch.long))
    loss_D.backward()
    optimizer_D.step()

    # === Train Generator ===
    optimizer_G.zero_grad()
    fake_seq = torch.argmax(model.generator(item_seq, lengths), dim=-1)
    disc_fake = model.discriminator(fake_seq)
    loss_G = F.cross_entropy(disc_fake, torch.ones(len(labels), dtype=torch.long))
    loss_G.backward()
    optimizer_G.step()

    return loss_D.item(), loss_G.item()


def train_step_gcl(batch, model, optimizer):
    """One training step for GCL module"""
    user_ids, item_seq, lengths, target_item, labels = batch

    optimizer.zero_grad()
    user_emb, item_emb = model.guig(user_ids, item_seq)

    # Positive: same user, negative: different users
    # (simplified - actual implementation needs proper sampling)
    h_u = user_emb
    h_u_pos = item_emb.mean(dim=1)  # simplified
    h_v_neg = user_emb.unsqueeze(1)  # simplified

    loss_gcl = model.gcl.contrastive_loss(h_u, h_u_pos, h_v_neg)
    loss_gcl.backward()
    optimizer.step()

    return loss_gcl.item()


def train_step_ctr(batch, model, optimizer):
    """One training step for CTR prediction"""
    user_ids, item_seq, lengths, target_item, labels = batch

    optimizer.zero_grad()
    logits = model(user_ids, item_seq, lengths, target_item)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    loss.backward()
    optimizer.step()

    return loss.item()
