"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning
Supports r∈{4,8,16}, α=r, dropout=0.05 as specified
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear layers
    
    Implements: h = W₀x + (B·A)x·(α/r) + dropout
    where W₀ is frozen, B∈R^(d×r), A∈R^(r×k), α is scaling factor
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights"""
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: h = W₀x + (B·A)x·(α/r) + dropout
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Original frozen forward pass
        h = self.linear(x)
        
        # LoRA adaptation
        # x: [..., in_features] -> [..., out_features]
        # Reshape to [batch_size, in_features] for matrix multiplication
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])  # [batch_size, in_features]
        
        # LoRA computation: B @ A @ x^T
        lora_out = self.lora_B @ (self.lora_A @ x_flat.transpose(-2, -1))  # [out_features, batch_size]
        lora_out = lora_out.transpose(-2, -1)  # [batch_size, out_features]
        lora_out = lora_out * self.scaling
        
        # Reshape back to original shape (except last dimension)
        lora_out = lora_out.view(*original_shape[:-1], -1)
        
        # Apply dropout and add to frozen output
        lora_out = self.dropout(lora_out)
        
        return h + lora_out
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for analysis"""
        frozen_params = sum(p.numel() for p in self.linear.parameters())
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        
        return {
            'frozen': frozen_params,
            'lora': lora_params,
            'total': frozen_params + lora_params,
            'trainable': lora_params
        }

def apply_lora_to_linear(
    module: nn.Module,
    target_modules: list = None,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05
) -> nn.Module:
    """
    Apply LoRA to specified linear layers in a module
    
    Args:
        module: PyTorch module to modify
        target_modules: List of module names to target (if None, target all Linear)
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout rate
        
    Returns:
        Modified module with LoRA applied
    """
    if target_modules is None:
        target_modules = ['Linear']
    
    def _replace_linear(module, name):
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, nn.Linear):
                # Replace with LoRA version
                lora_linear = LoRALinear(
                    in_features=attr.in_features,
                    out_features=attr.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=attr.bias is not None
                )
                # Copy original weights
                lora_linear.linear.weight.data = attr.weight.data.clone()
                if attr.bias is not None:
                    lora_linear.linear.bias.data = attr.bias.data.clone()
                
                setattr(module, attr_name, lora_linear)
        
        # Recursively apply to child modules
        for child_name, child_module in module.named_children():
            _replace_linear(child_module, child_name)
    
    _replace_linear(module, 'root')
    return module

def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count LoRA parameters in a model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0
    frozen_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            counts = module.get_parameter_count()
            lora_params += counts['lora']
            frozen_params += counts['frozen']
        elif hasattr(module, 'weight'):
            if module.weight.requires_grad:
                trainable_params += module.weight.numel()
            else:
                frozen_params += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                if module.bias.requires_grad:
                    trainable_params += module.bias.numel()
                else:
                    frozen_params += module.bias.numel()
    
    total_params = trainable_params + frozen_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'lora': lora_params,
        'lora_ratio': lora_params / total_params if total_params > 0 else 0.0
    }

def create_lora_config(rank: int = 4, alpha: float = 8.0, dropout: float = 0.05) -> Dict[str, Any]:
    """Create LoRA configuration dictionary"""
    return {
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'scaling': alpha / rank,
        'target_modules': ['Linear', 'Conv1d', 'Conv2d']  # Can be extended
    }
