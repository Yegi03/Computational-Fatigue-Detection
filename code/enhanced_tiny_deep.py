"""
Enhanced Tiny-Deep Model with LoRA and/or MoE
Supports: LoRA-only, MoE-only, LoRA+MoE, or Full Fine-tuning
Parameter-efficient fine-tuning for mental fatigue detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from .lora import LoRALinear, apply_lora_to_linear, count_lora_parameters
from .moe import MixtureOfExperts

class EEGEncoderLoRA(nn.Module):
    """EEG encoder with optional LoRA adaptation"""
    def __init__(self, input_dim: int, output_dim: int = 128, lora_rank: int = 4, use_lora: bool = True):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        if use_lora:
        self.linear = LoRALinear(64, output_dim, rank=lora_rank)
        else:
            self.linear = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1)  # [B, 64]
        return self.linear(x)  # [B, output_dim]

class ECGHRVEncoderLoRA(nn.Module):
    """ECG-HRV encoder with optional LoRA adaptation"""
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_size: int = 64, 
                 num_layers: int = 2, lora_rank: int = 4, use_lora: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=0.1)
        if use_lora:
        self.linear = LoRALinear(hidden_size, output_dim, rank=lora_rank)
        else:
            self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: [num_layers, B, hidden_size]
        return self.linear(h_n[-1])  # [B, output_dim]

class GSREncoderLoRA(nn.Module):
    """GSR encoder with optional LoRA adaptation"""
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_size: int = 32, 
                 lora_rank: int = 4, use_lora: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, 1, batch_first=True, dropout=0.0)
        if use_lora:
        self.linear = LoRALinear(hidden_size, output_dim, rank=lora_rank)
        else:
            self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: [1, B, hidden_size]
        return self.linear(h_n[-1])  # [B, output_dim]

class EnhancedTinyDeepFatigueNet(nn.Module):
    """
    Enhanced Tiny-Deep Model with LoRA and/or MoE
    
    Features:
    - Optional LoRA adaptation on all linear layers
    - Optional MoE (Mixture of Experts) routing
    - Multi-task learning (PFI + ToT + ACL)
    - Parameter counting and efficiency analysis
    
    Configurations:
    - LoRA-only: use_lora=True, use_moe=False
    - MoE-only: use_lora=False, use_moe=True
    - LoRA+MoE: use_lora=True, use_moe=True
    - Full Fine-tuning: use_lora=False, use_moe=False (all params trainable)
    """
    
    def __init__(
        self,
        eeg_input_dim: int,
        ecg_input_dim: int,
        gsr_input_dim: int,
        eeg_seq_len: int = 10,
        ecg_seq_len: int = 1,
        gsr_seq_len: int = 10,
        fusion_dim: int = 256,
        encoder_output_dim: int = 128,
        lora_rank: int = 16,
        use_lora: bool = True,
        use_moe: bool = False,
        moe_experts: int = 3,
        use_personality: bool = True,
        gate_type: str = 'soft',
        loss_weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.use_lora = use_lora
        self.use_moe = use_moe
        self.use_personality = use_personality
        
        # Encoders with optional LoRA
        self.eeg_encoder = EEGEncoderLoRA(
            eeg_input_dim * eeg_seq_len, 
            encoder_output_dim, 
            lora_rank,
            use_lora=use_lora
        )
        self.ecg_encoder = ECGHRVEncoderLoRA(
            ecg_input_dim, 
            encoder_output_dim, 
            hidden_size=64, 
            num_layers=2,
            lora_rank=lora_rank,
            use_lora=use_lora
        )
        self.gsr_encoder = GSREncoderLoRA(
            gsr_input_dim, 
            encoder_output_dim, 
            hidden_size=32,
            lora_rank=lora_rank,
            use_lora=use_lora
        )
        
        # Fusion layer with optional LoRA
        if use_lora:
        self.fusion = nn.Sequential(
            LoRALinear(encoder_output_dim * 3, fusion_dim, rank=lora_rank),
            nn.ReLU(),
            nn.Dropout(0.1),
            LoRALinear(fusion_dim, fusion_dim, rank=lora_rank),
            nn.LayerNorm(fusion_dim)
        )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(encoder_output_dim * 3, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim)
            )
        
        # MoE routing (optional)
        if use_moe:
            self.moe = MixtureOfExperts(
                input_dim=fusion_dim,
                output_dim=fusion_dim,
                num_experts=moe_experts,
                expert_hidden_dim=128,
                expert_num_layers=2,
                personality_dim=5,
                use_personality=use_personality,
                gate_type=gate_type,
                temperature=1.0,
                entropy_reg=0.02
            )
        else:
            self.moe = None
        
        # Multi-task heads with optional LoRA
        if use_lora:
        self.pfi_head = nn.Sequential(
            LoRALinear(fusion_dim, 64, rank=lora_rank),
            nn.ReLU(),
            LoRALinear(64, 1, rank=lora_rank)
        )
        self.tot_head = nn.Sequential(
            LoRALinear(fusion_dim, 64, rank=lora_rank),
            nn.ReLU(),
            LoRALinear(64, 1, rank=lora_rank)
        )
        self.acl_head = nn.Sequential(
            LoRALinear(fusion_dim, 64, rank=lora_rank),
            nn.ReLU(),
            LoRALinear(64, 1, rank=lora_rank)
        )
        else:
            self.pfi_head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.tot_head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.acl_head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        self.loss_weights = loss_weights or {
            'pfi': 0.6,
            'tot': 0.2,
            'acl': 0.2
        }
        
        # Initialize weights first
        self._init_weights()
        
        # Freeze backbone if using LoRA or MoE (parameter-efficient)
        # Must be done after initialization so all parameters exist
        if use_lora or use_moe:
            # Freeze all non-LoRA/MoE parameters
            for name, param in self.named_parameters():
                if 'lora' not in name.lower() and 'moe' not in name.lower() and 'gate' not in name.lower():
                    param.requires_grad = False
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        ecg_features: torch.Tensor,
        gsr_features: torch.Tensor,
        modality_mask: torch.Tensor,
        personality: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through enhanced model
        
        Args:
            eeg_features: [B, T_eeg, F_eeg]
            ecg_features: [B, T_hrv, F_hrv]
            gsr_features: [B, T_gsr, F_gsr]
            modality_mask: [B, 3]
            personality: [B, 5] (optional)
            
        Returns:
            Dictionary with outputs and statistics
        """
        B = eeg_features.shape[0]
        
        # Flatten EEG features for 1D-CNN
        eeg_flat = eeg_features.view(B, -1)  # [B, T_eeg * F_eeg]
        
        # Encode modalities
        eeg_encoded = self.eeg_encoder(eeg_flat)  # [B, encoder_output_dim]
        ecg_encoded = self.ecg_encoder(ecg_features)  # [B, encoder_output_dim]
        gsr_encoded = self.gsr_encoder(gsr_features)  # [B, encoder_output_dim]
        
        # Apply modality masking
        eeg_encoded = eeg_encoded * modality_mask[:, 0:1]
        ecg_encoded = ecg_encoded * modality_mask[:, 1:2]
        gsr_encoded = gsr_encoded * modality_mask[:, 2:3]
        
        # Fuse modalities
        fused_features = torch.cat([eeg_encoded, ecg_encoded, gsr_encoded], dim=1)
        z = self.fusion(fused_features)  # [B, fusion_dim]
        
        # MoE routing (optional)
        if self.moe is not None:
            moe_output = self.moe(z, personality)
            z_star = moe_output['output']
            gate_entropy = moe_output['entropy']
            gate_weights = moe_output['gate_weights']
        else:
            z_star = z
            gate_entropy = torch.tensor(0.0, device=z.device)
            gate_weights = None
        
        # Multi-task heads
        pfi_pred = self.pfi_head(z_star).squeeze(-1)  # [B]
        tot_logits = self.tot_head(z_star).squeeze(-1)  # [B]
        acl_logits = self.acl_head(z_star).squeeze(-1)  # [B]
        
        return {
            'pfi_pred': pfi_pred,
            'tot_logits': tot_logits,
            'acl_logits': acl_logits,
            'tot_prob': torch.sigmoid(tot_logits),
            'acl_prob': torch.sigmoid(acl_logits),
            'fusion_embedding': z_star,
            'gate_entropy': gate_entropy,
            'gate_weights': gate_weights
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with entropy regularization"""
        losses = {}
        
        # PFI Regression Loss
        if 'pfi' in targets:
            pfi_mask = ~torch.isnan(targets['pfi'])
            if pfi_mask.any():
                losses['pfi_loss'] = F.mse_loss(outputs['pfi_pred'][pfi_mask], targets['pfi'][pfi_mask])
            else:
                losses['pfi_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        else:
            losses['pfi_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        
        # ToT Classification Loss
        if 'tot' in targets:
            tot_mask = ~torch.isnan(targets['tot'])
            if tot_mask.any():
                losses['tot_loss'] = F.binary_cross_entropy_with_logits(
                    outputs['tot_logits'][tot_mask], targets['tot'][tot_mask].float()
                )
            else:
                losses['tot_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        else:
            losses['tot_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        
        # ACL Classification Loss
        if 'acl' in targets:
            acl_mask = ~torch.isnan(targets['acl'])
            if acl_mask.any():
                losses['acl_loss'] = F.binary_cross_entropy_with_logits(
                    outputs['acl_logits'][acl_mask], targets['acl'][acl_mask].float()
                )
            else:
                losses['acl_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        else:
            losses['acl_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        
        # Entropy regularization (if MoE is used)
        if self.moe is not None:
            losses['entropy_loss'] = -outputs['gate_entropy']  # Negative for regularization
        else:
            losses['entropy_loss'] = torch.tensor(0.0, device=outputs['pfi_pred'].device)
        
        # Total loss
        total_loss = (
            self.loss_weights.get('pfi', 0.0) * losses['pfi_loss'] +
            self.loss_weights.get('tot', 0.0) * losses['tot_loss'] +
            self.loss_weights.get('acl', 0.0) * losses['acl_loss'] +
            0.02 * losses['entropy_loss']  # Entropy regularization weight
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def get_parameter_analysis(self) -> Dict[str, Any]:
        """Get comprehensive parameter analysis"""
        lora_counts = count_lora_parameters(self)
        
        analysis = {
            'model_name': self._get_model_name(),
            'total_parameters': lora_counts['total'],
            'trainable_parameters': lora_counts['trainable'],
            'frozen_parameters': lora_counts['frozen'],
            'lora_parameters': lora_counts['lora'],
            'lora_ratio': lora_counts['lora_ratio'],
            'use_lora': self.use_lora,
            'use_moe': self.use_moe,
            'use_personality': self.use_personality
        }
        
        if self.moe is not None:
            moe_counts = self.moe.get_parameter_count()
            analysis.update({
                'moe_parameters': moe_counts['total'],
                'moe_experts': self.moe.num_experts,
                'moe_gate_type': self.moe.gate.gate_type
            })
        
        return analysis
    
    def _get_model_name(self) -> str:
        """Get model name based on configuration"""
        if self.use_lora and self.use_moe:
            return 'Enhanced Tiny-Deep (LoRA + MoE)'
        elif self.use_lora:
            return 'Enhanced Tiny-Deep (LoRA only)'
        elif self.use_moe:
            return 'Enhanced Tiny-Deep (MoE only)'
        else:
            return 'Enhanced Tiny-Deep (Full Fine-tuning)'
    
    def get_expert_usage(self) -> Dict[str, float]:
        """Get MoE expert usage statistics"""
        if self.moe is not None:
            return self.moe.get_expert_usage()
        else:
            return {}
    
    def reset_expert_usage(self):
        """Reset MoE expert usage statistics"""
        if self.moe is not None:
            self.moe.reset_usage_stats()

def create_enhanced_model_config(
    lora_rank: int = 16,
    use_lora: bool = True,
    use_moe: bool = False,
    moe_experts: int = 3,
    use_personality: bool = True,
    gate_type: str = 'soft'
) -> Dict[str, Any]:
    """Create enhanced model configuration"""
    return {
        'lora_rank': lora_rank,
        'use_lora': use_lora,
        'use_moe': use_moe,
        'moe_experts': moe_experts,
        'use_personality': use_personality,
        'gate_type': gate_type,
        'fusion_dim': 256,
        'encoder_output_dim': 128
    }
