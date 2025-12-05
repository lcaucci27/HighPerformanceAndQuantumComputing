"""
Custom activation functions
SQNL (Squared Non-Linearity) from the paper
"""

import torch
import torch.nn as nn

class SQNL(nn.Module):
    """
    Squared Non-Linearity (SQNL) activation function
    
    SQNL(x) = {
        -1           if x < -1
        2x + x²      if -1 ≤ x < 0
        2x - x²      if 0 ≤ x ≤ 1
        1            if x > 1
    }
    
    Hardware-friendly alternative to tanh with similar performance
    """
    
    def __init__(self):
        super(SQNL, self).__init__()
    
    def forward(self, x):
        """
        Apply SQNL activation element-wise
        
        Args:
            x: Input tensor
        
        Returns:
            Activated tensor
        """
        # Create output tensor
        output = torch.zeros_like(x)
        
        # Case 1: x < -1 → -1
        mask1 = x < -1
        output[mask1] = -1.0
        
        # Case 2: -1 ≤ x < 0 → 2x + x²
        mask2 = (x >= -1) & (x < 0)
        output[mask2] = 2 * x[mask2] + x[mask2] ** 2
        
        # Case 3: 0 ≤ x ≤ 1 → 2x - x²
        mask3 = (x >= 0) & (x <= 1)
        output[mask3] = 2 * x[mask3] - x[mask3] ** 2
        
        # Case 4: x > 1 → 1
        mask4 = x > 1
        output[mask4] = 1.0
        
        return output


class SQNLEfficient(nn.Module):
    """
    More efficient SQNL implementation using torch.where
    """
    
    def __init__(self):
        super(SQNLEfficient, self).__init__()
    
    def forward(self, x):
        """Apply SQNL activation using nested torch.where"""
        return torch.where(
            x < -1, 
            -torch.ones_like(x),
            torch.where(
                x < 0, 
                2*x + x**2,
                torch.where(
                    x <= 1, 
                    2*x - x**2, 
                    torch.ones_like(x)
                )
            )
        )