import math
import torch
import torch.nn as nn
from kan_layers import MLPFFN, KANFFN


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for transformer models.
    """
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                       (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input token embeddings [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Position-aware embeddings [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """
    A single transformer block with configurable FFN type (MLP or KAN).
    """
    def __init__(self, d_model, n_heads, d_ff,
                 use_kan_ffn=False, kan_grid=3, kan_width_factor=0.03125,
                 attn_dropout=0.1, resid_dropout=0.1):
        """
        Args:
            d_model (int): Dimension of token embeddings
            n_heads (int): Number of attention heads
            d_ff (int): Base feed-forward dimension
            use_kan_ffn (bool): If True, use KAN-based FFN; else use standard MLP
            kan_grid (int): Grid size for KAN spline approximation
            kan_width_factor (float): Width reduction factor for KAN 
            attn_dropout (float): Dropout rate for attention outputs
            resid_dropout (float): Dropout rate for residual connections
        """
        super().__init__()
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        
        # For KAN, use reduced width
        if use_kan_ffn:
            self.ffn = KANFFN(
                d_model, 
                d_ff, 
                grid_size=kan_grid,
                kan_width_factor=kan_width_factor
            )
        # For MLP, use full width (standard transformer)
        else:
            self.ffn = MLPFFN(d_model, d_ff)
        
        # Dropout layers for residual connections
        self.drop1 = nn.Dropout(resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d_model]
        """
        # Attention layer with residual connection
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        
        # Feed-forward with residual connection
        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        
        return x


class MiniTransformerLM(nn.Module):
    """
    A minimal transformer language model with configurable FFN type.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, block_size,
                 use_kan_ffn=False, kan_grid=3, kan_width_factor=0.03125):
        """
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of token embeddings
            n_heads (int): Number of attention heads in each block
            n_layers (int): Number of transformer blocks
            d_ff (int): Base feed-forward dimension
            block_size (int): Maximum sequence length (context window)
            use_kan_ffn (bool): If True, use KAN-based FFNs; else use MLP FFNs
            kan_grid (int): Grid size for KAN spline approximation
            kan_width_factor (float): Width reduction factor for KAN (0.1-0.5)
        """
        super().__init__()
        
        self.block_size = block_size
        
        # Token embedding layer
        self.tok = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos = PositionalEncoding(d_model, max_len=block_size + 8)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, 
                n_heads, 
                d_ff,
                use_kan_ffn=use_kan_ffn,
                kan_grid=kan_grid,
                kan_width_factor=kan_width_factor
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization and projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init)

    def _init(self, m):
        """
        Weight initialization.
        
        Args:
            m (nn.Module): A submodule of the transformer
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        """    
        Args:
            idx (torch.Tensor): Input token indices [batch_size, seq_len]
            targets (torch.Tensor, optional): Target token indices for loss computation
            
        Returns:
            tuple: (logits, loss) where:
                - logits: [batch_size, seq_len, vocab_size] prediction scores
                - loss: scalar loss value (None if targets is None)
        """
        # Convert token indices to embeddings
        x = self.tok(idx)
        
        # Add positional information
        x = self.pos(x)
        
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final layer normalization and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        
        return logits, loss


def count_params(m: nn.Module):
    """
    Args:
        m (nn.Module): PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)