"""
SeqUGA: Sequential User Behavior Modeling with Ultra-Long History
Based on paper: http://arxiv.org/abs/2509.17361v1

Paper highlights:
- Problem: Ultra-long user behavior sequences (10000+) degrade recommendation
- Solution: Sequence Split + Graph Aggregation to handle length efficiently
- Result: O(n) complexity vs O(n²) for Transformer-based methods

Optimized by darwin-skill (v1.4)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


# ============== Part 1: Sequence Split & Graph Construction ==============

class SequenceSplitter:
    """
    Split long sequence into subsequences for graph construction.
    From paper: S = [b1, b2, ..., bn] → [S1, S2, ..., Sk]
    """

    def __init__(self, subsequence_length: int = 50):
        self.subsequence_length = subsequence_length

    def split(self, item_sequence: list) -> list:
        """
        Split long sequence into subsequences.

        Args:
            item_sequence: list of item IDs (can be 10000+ length)

        Returns:
            subsequences: list of subsequence lists
        """
        subsequences = []
        for i in range(0, len(item_sequence), self.subsequence_length):
            subseq = item_sequence[i:i + self.subsequence_length]
            if len(subseq) >= 2:  # Skip very short subsequences
                subsequences.append(subseq)
        return subsequences


class BehaviorGraphBuilder:
    """
    Build behavior graph from subsequences.
    Edges connect adjacent items within and across subsequences.
    """

    def __init__(self):
        self.edges = []

    def build(self, subsequences: list):
        """
        Build edge list from subsequences.

        Args:
            subsequences: list of subsequence lists

        Returns:
            edge_index: (2, num_edges) tensor for graph convolution
        """
        edges = []
        node_map = {}  # (subseq_idx, pos) -> global_node_id

        # Assign global node IDs
        node_id = 0
        for seq_idx, subseq in enumerate(subsequences):
            for pos, item_id in enumerate(subseq):
                node_map[(seq_idx, pos)] = node_id
                node_id += 1

        # Build edges within each subsequence
        for seq_idx, subseq in enumerate(subsequences):
            for pos in range(len(subseq) - 1):
                src = node_map[(seq_idx, pos)]
                dst = node_map[(seq_idx, pos + 1)]
                edges.append([src, dst])
                edges.append([dst, src])  # Undirected

        # Build edges between consecutive subsequences (bridge items)
        for i in range(len(subsequences) - 1):
            last_item = subsequences[i][-1]
            first_item = subsequences[i + 1][0]
            if last_item == first_item:  # Same item bridges subsequences
                src = node_map[(i, len(subsequences[i]) - 1)]
                dst = node_map[(i + 1, 0)]
                edges.append([src, dst])
                edges.append([dst, src])

        self.edges = edges
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ============== Part 2: Graph Aggregation Layer ==============

class GraphAggregationLayer(nn.Module):
    """
    Graph Attention Layer from paper Equation (2):
        hi = Attention(Qi, Kj, Vj) for j in neighbors(i)

    Complexity: O(n) instead of O(n²) for Transformer attention
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN (from paper: GELU activation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: (num_nodes, embed_dim) node embeddings
            edge_index: (2, num_edges) edge indices

        Returns:
            output: (num_nodes, embed_dim) updated embeddings
        """
        batch_size, num_nodes, _ = x.shape
        x = x.view(-1, self.embed_dim)  # Flatten for graph conv

        # Compute Q, K, V
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        # Simple graph attention: aggregate from neighbors
        # For efficiency, we use a simplified version without sparse operations
        out = self._graph_attention(q, k, v, edge_index, num_nodes)

        # FFN + Residual
        out = self.ffn(out) + out
        out = self.norm2(out)

        return out.view(batch_size, num_nodes, -1)

    def _graph_attention(self, q, k, v, edge_index, num_nodes):
        """Simplified graph attention operation."""
        # Expand edge_index for heads
        src, dst = edge_index[0], edge_index[1]

        # Compute attention scores
        attn_scores = (q[src] * k[dst]).sum(dim=-1)  # (num_edges, num_heads)

        # Softmax over neighbors (simplified - no mask)
        attn_weights = torch.softmax(attn_scores, dim=0)

        # Aggregate values
        num_edges = src.shape[0]
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=q.device)

        for i in range(num_edges):
            out[dst[i]] += attn_weights[i].unsqueeze(-1) * v[src[i]]

        out = out.reshape(num_nodes, self.embed_dim)
        return self.out_proj(out)


# ============== Part 3: SeqUGA Full Model ==============

class SeqUGA(nn.Module):
    """
    SeqUGA: Sequential User Behavior Modeling with Ultra-Long History

    Architecture:
    1. Sequence Split: Split long sequence into subsequences
    2. Graph Build: Build behavior graph from subsequences
    3. Graph Aggregation: Process with Graph Attention layers
    4. Output: Unified item embeddings for recommendation
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 num_layers: int = 3, num_heads: int = 4,
                 subsequence_length: int = 50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Graph aggregation layers
        self.layers = nn.ModuleList([
            GraphAggregationLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output = nn.Linear(embed_dim, vocab_size)

        # Auxiliary modules
        self.splitter = SequenceSplitter(subsequence_length)
        self.graph_builder = BehaviorGraphBuilder()

    def forward(self, item_sequence: torch.Tensor):
        """
        Args:
            item_sequence: (batch_size, seq_len) item IDs

        Returns:
            logits: (batch_size, vocab_size) next-item prediction
        """
        batch_size, seq_len = item_sequence.shape

        # Convert to list for processing
        sequences = item_sequence.cpu().numpy().tolist()

        outputs = []
        for seq in sequences:
            out = self._process_sequence(seq)
            outputs.append(out)

        # Stack results
        logits = torch.stack(outputs, dim=0)  # (batch_size, vocab_size)
        return logits

    def _process_sequence(self, item_seq: list):
        """Process single long sequence."""
        # Step 1: Split into subsequences
        subsequences = self.splitter.split(item_seq)

        if len(subsequences) == 0:
            # Very short sequence, use simple embedding
            x = self.embedding(torch.tensor(item_seq, device=next(self.parameters()).device))
            x = x.mean(dim=0, keepdim=True)
        else:
            # Step 2: Build graph
            edge_index = self.graph_builder.build(subsequences)

            # Step 3: Create node features
            num_nodes = sum(len(s) for s in subsequences)
            node_items = [item for subseq in subsequences for item in subseq]
            x = self.embedding(torch.tensor(node_items, device=next(self.parameters()).device))

            # Step 4: Graph aggregation
            x = x.unsqueeze(0)  # (1, num_nodes, embed_dim)
            edge_index = edge_index.to(next(self.parameters()).device)

            for layer in self.layers:
                x = layer(x, edge_index)

            x = x.squeeze(0)  # (num_nodes, embed_dim)

            # Step 5: Pool to get sequence representation
            x = x.mean(dim=0)  # (embed_dim,)

        # Step 6: Output projection
        logits = self.output(x)  # (vocab_size,)
        return logits


# ============== Part 4: Demo Usage ==============

def demo():
    """Demo: Run SeqUGA on synthetic sequence data."""

    # Hyperparameters from paper
    VOCAB_SIZE = 10000
    EMBED_DIM = 64
    NUM_LAYERS = 3
    NUM_HEADS = 4
    SUBSEQ_LENGTH = 50

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SeqUGA(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        subsequence_length=SUBSEQ_LENGTH
    ).to(device)

    # Synthetic data: batch of 4 sequences with varying lengths
    sequences = [
        [101, 203, 1547, 203, 889, 2341, 189, 3421, 776, 1547] * 50,  # 500 items
        [2341, 189, 3421, 776] * 100,  # 400 items
        [101, 203, 1547] * 200,  # 600 items
        [889, 2341, 189, 3421, 776, 1547, 203] * 80,  # 560 items
    ]
    batch = torch.tensor(sequences, dtype=torch.long).to(device)

    print("=" * 60)
    print("SeqUGA: Ultra-Long Sequence Modeling with Graph Aggregation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: Vocab={VOCAB_SIZE}, Dim={EMBED_DIM}, Layers={NUM_LAYERS}")
    print(f"Input: {len(sequences)} sequences, lengths={[len(s) for s in sequences]}")
    print()

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(batch)

    print(f"Output shape: {logits.shape}")
    print(f"Logit range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Top-k predictions
    _, topk = torch.topk(logits, k=5, dim=-1)
    print(f"\nTop-5 predictions for first sequence:")
    print(f"  Items: {topk[0].tolist()}")

    print()
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()