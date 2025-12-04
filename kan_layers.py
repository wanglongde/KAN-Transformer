import torch
import torch.nn as nn
from pykan.kan import KANLayer


class MLPFFN(nn.Module):
    """
    Standard Multi-Layer Perceptron Feed-Forward Network (Baseline)
    
    This is the conventional FFN used in transformers with two linear layers and a fixed ReLU activation function. 
    Serves as the baseline comparison for the KAN-based FFN.
    """
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model (int): Dimension of the model (input and output size).
            d_ff (int): Dimension of the hidden layer in the feed-forward network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  
            nn.ReLU(),                 
            nn.Linear(d_ff, d_model),  
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor of same shape [batch_size, seq_len, d_model]
        """
        return self.net(x)


class KANFFN(nn.Module):
    """
    Kolmogorov-Arnold Network Feed-Forward Network
    
    Implements the two-layer KANs with much smaller widths than MLPs
    to achieve comparable parameter counts and better generalization.
    """
    def __init__(self, d_model, d_ff, grid_size=3, kan_width_factor=0.03125):
        """ 
        Args:
            d_model (int): Dimension of the model (input and output size)
            d_ff (int): Base feed-forward dimension (from transformer config)
            grid_size (int): Number of grid intervals for spline approximation
            kan_width_factor (float): Reduction factor for KAN hidden dimension (0.1-0.5)
                                    Paper suggests KAN needs much smaller width than MLP
        """
        super().__init__()
        
        # KANs require much smaller width than MLPs
        # Calculate reduced hidden dimension for KAN
        kan_hidden = max(1, int(d_ff * kan_width_factor))
        
        print(f"[KANFFN] d_model={d_model}, d_ff={d_ff} -> kan_hidden={kan_hidden} (factor={kan_width_factor})")
        
        self.fc1 = KANLayer(
            in_dim=d_model,
            out_dim=kan_hidden, 
            num=grid_size,
            k=3,
            grid_range=[-1, 1],
            base_fun=nn.SiLU()
        )
         
        self.fc2 = KANLayer(
            in_dim=kan_hidden,
            out_dim=d_model,
            num=grid_size,
            k=3,
            grid_range=[-1, 1],
            base_fun=nn.SiLU()
        )
        
        self.d_model = d_model
        self.kan_hidden = kan_hidden

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor of same shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape from 3D to 2D for KANLayer
        x_flat = x.reshape(-1, d_model)
        
        # First KAN layer: [batch_size * seq_len, d_model] -> [batch_size * seq_len, kan_hidden]
        y1, _, _, _ = self.fc1(x_flat)
        
        # Second KAN layer: [batch_size * seq_len, kan_hidden] -> [batch_size * seq_len, d_model]
        y2, _, _, _ = self.fc2(y1)
        
        # Reshape from 2D to 3D
        y2 = y2.reshape(batch_size, seq_len, self.d_model)
        
        return y2


def count_params(m: nn.Module):
    """
    Args:
        m (nn.Module): PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)