"""
Mixture of Experts (MoE) implementation
K∈{1,3,4} expert networks with hard/soft gating, personality input ablation

MoE uses full expert networks (not bottleneck adapters).
Each expert is a complete MLP that processes the input independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

class ExpertNetwork(nn.Module):
    """
    Single expert network in the mixture (MoE pattern)
    
    This is a full expert network: input_dim → hidden_dim → hidden_dim → output_dim
    Each expert processes the input independently (no residual connection).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Build expert network layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Add intermediate layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.expert = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize expert network weights"""
        for module in self.expert:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network"""
        return self.expert(x)

class MoEGate(nn.Module):
    """Gating network for MoE with optional personality input"""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        personality_dim: int = 5,
        use_personality: bool = True,
        gate_type: str = 'hard',  # 'hard' or 'soft'
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.gate_type = gate_type
        self.temperature = temperature
        self.use_personality = use_personality
        
        # Input dimension includes personality if used
        gate_input_dim = input_dim + (personality_dim if use_personality else 0)
        
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gate weights"""
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        personality: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through gating network
        
        Args:
            x: Input features [B, input_dim]
            personality: Personality features [B, personality_dim] (optional)
            
        Returns:
            Gate weights [B, num_experts]
        """
        # Concatenate personality if provided
        if self.use_personality and personality is not None:
            gate_input = torch.cat([x, personality], dim=-1)
        else:
            gate_input = x
        
        # Compute gate logits
        gate_logits = self.gate_network(gate_input) / self.temperature
        
        if self.gate_type == 'hard':
            # Hard gating: select top-1 expert
            gate_weights = F.one_hot(
                torch.argmax(gate_logits, dim=-1), 
                num_classes=self.num_experts
            ).float()
        else:
            # Soft gating: softmax over experts
            gate_weights = F.softmax(gate_logits, dim=-1)
        
        return gate_weights

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) with K expert networks and gating
    
    Supports:
    - K∈{1,3,4} expert networks
    - Hard/soft gating
    - Personality input ablation
    - Entropy regularization for diversity
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 3,
        expert_hidden_dim: int = 128,
        expert_num_layers: int = 2,
        personality_dim: int = 5,
        use_personality: bool = True,
        gate_type: str = 'hard',
        temperature: float = 1.0,
        entropy_reg: float = 0.02,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.entropy_reg = entropy_reg
        
        # Create expert networks (full networks, not adapters)
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=expert_hidden_dim,
                num_layers=expert_num_layers,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gate = MoEGate(
            input_dim=input_dim,
            num_experts=num_experts,
            personality_dim=personality_dim,
            use_personality=use_personality,
            gate_type=gate_type,
            temperature=temperature
        )
        
        # Statistics tracking
        self.gate_usage = torch.zeros(num_experts)
        self.forward_count = 0
    
    def forward(
        self, 
        x: torch.Tensor, 
        personality: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE
        
        Args:
            x: Input features [B, input_dim]
            personality: Personality features [B, personality_dim] (optional)
            
        Returns:
            Dictionary with outputs and statistics
        """
        batch_size = x.shape[0]
        
        # Compute gate weights
        gate_weights = self.gate(x, personality)  # [B, num_experts]
        
        # Compute expert outputs (each expert processes input independently)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [B, output_dim]
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, output_dim]
        
        # Weighted combination
        if self.gate.gate_type == 'hard':
            # Hard gating: select expert outputs directly
            expert_indices = torch.argmax(gate_weights, dim=-1)  # [B]
            output = expert_outputs[torch.arange(batch_size), expert_indices]  # [B, output_dim]
        else:
            # Soft gating: weighted sum
            output = torch.sum(
                gate_weights.unsqueeze(-1) * expert_outputs, 
                dim=1
            )  # [B, output_dim]
        
        # Compute entropy for regularization
        gate_probs = F.softmax(gate_weights, dim=-1)
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1).mean()
        
        # Update usage statistics
        if self.training:
            self.gate_usage += gate_weights.sum(dim=0).detach()
            self.forward_count += batch_size
        
        return {
            'output': output,
            'gate_weights': gate_weights,
            'entropy': entropy,
            'expert_outputs': expert_outputs
        }
    
    def get_expert_usage(self) -> Dict[str, float]:
        """Get expert usage statistics"""
        if self.forward_count == 0:
            return {f'expert_{i}': 0.0 for i in range(self.num_experts)}
        
        usage = self.gate_usage / self.forward_count
        return {f'expert_{i}': float(usage[i]) for i in range(self.num_experts)}
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.gate_usage.zero_()
        self.forward_count = 0
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for analysis"""
        expert_params = sum(sum(p.numel() for p in expert.parameters()) for expert in self.experts)
        gate_params = sum(p.numel() for p in self.gate.parameters())
        
        return {
            'experts': expert_params,
            'gate': gate_params,
            'total': expert_params + gate_params
        }

# For backward compatibility, you can import as:
# But the primary interface is MixtureOfExperts

def create_moe_config(
    num_experts: int = 3,
    expert_hidden_dim: int = 128,
    expert_num_layers: int = 2,
    use_personality: bool = True,
    gate_type: str = 'hard',
    temperature: float = 1.0,
    entropy_reg: float = 0.02
) -> Dict[str, Any]:
    """Create MoE configuration dictionary"""
    return {
        'num_experts': num_experts,
        'expert_hidden_dim': expert_hidden_dim,
        'expert_num_layers': expert_num_layers,
        'use_personality': use_personality,
        'gate_type': gate_type,
        'temperature': temperature,
        'entropy_reg': entropy_reg
    }

# Use create_moe_config for new code
