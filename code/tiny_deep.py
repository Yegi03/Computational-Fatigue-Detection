#!/usr/bin/env python3
"""
Tiny-Deep Baseline Model for Mental Fatigue Detection
A lightweight neural network without adapters for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class EEGEncoder(nn.Module):
    """1D-CNN encoder for EEG features"""
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [B, T, F_eeg] -> [B, 1, T*F_eeg] for 1D-CNN
        x = x.unsqueeze(1) # [B, 1, input_dim]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x).squeeze(-1) # [B, 64]
        return self.linear(x) # [B, output_dim]

class ECGHRVEncoder(nn.Module):
    """GRU encoder for ECG-HRV features"""
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: [B, T_hrv, F_hrv]
        _, h_n = self.gru(x) # h_n: [num_layers, B, hidden_size]
        return self.linear(h_n[-1]) # [B, output_dim]

class GSREncoder(nn.Module):
    """GRU encoder for GSR features"""
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, 1, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: [B, T_gsr, F_gsr]
        _, h_n = self.gru(x) # h_n: [1, B, hidden_size]
        return self.linear(h_n[-1]) # [B, output_dim]

class TinyDeepBaseline(nn.Module):
    """
    Tiny-Deep Baseline Model for mental fatigue detection.
    Uses 1D-CNN for EEG, GRUs for ECG-HRV and GSR, followed by a fusion MLP and multi-task heads.
    """
    def __init__(
        self,
        eeg_input_dim: int, # flattened feature dim for 10s window
        ecg_input_dim: int, # feature dim per 60s HRV frame
        gsr_input_dim: int, # feature dim per 10s GSR frame
        eeg_seq_len: int = 10, # Number of 10s EEG frames
        ecg_seq_len: int = 1, # Number of 60s HRV frames (aligned to 10s window)
        gsr_seq_len: int = 10, # Number of 10s GSR frames
        fusion_dim: int = 256,
        encoder_output_dim: int = 128,
        loss_weights: Dict[str, float] = None
    ):
        super().__init__()

        self.eeg_encoder = EEGEncoder(eeg_input_dim * eeg_seq_len, encoder_output_dim)
        self.ecg_encoder = ECGHRVEncoder(ecg_input_dim, encoder_output_dim, hidden_size=64, num_layers=2)
        self.gsr_encoder = GSREncoder(gsr_input_dim, encoder_output_dim, hidden_size=32)

        self.fusion = nn.Sequential(
            nn.Linear(encoder_output_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Multi-task heads
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(
        self,
        eeg_features: torch.Tensor, # [B, T_eeg, F_eeg]
        ecg_features: torch.Tensor, # [B, T_hrv, F_hrv]
        gsr_features: torch.Tensor, # [B, T_gsr, F_gsr]
        modality_mask: torch.Tensor, # [B, 3]
        personality: torch.Tensor = None # [B, 5] - not used in this baseline
    ) -> Dict[str, torch.Tensor]:
        
        B = eeg_features.shape[0]

        # Flatten EEG features for 1D-CNN
        eeg_flat = eeg_features.view(B, -1) # [B, T_eeg * F_eeg]

        eeg_encoded = self.eeg_encoder(eeg_flat) # [B, encoder_output_dim]
        ecg_encoded = self.ecg_encoder(ecg_features) # [B, encoder_output_dim]
        gsr_encoded = self.gsr_encoder(gsr_features) # [B, encoder_output_dim]

        # Apply modality masking
        eeg_encoded = eeg_encoded * modality_mask[:, 0:1]
        ecg_encoded = ecg_encoded * modality_mask[:, 1:2]
        gsr_encoded = gsr_encoded * modality_mask[:, 2:3]

        fused_features = torch.cat([eeg_encoded, ecg_encoded, gsr_encoded], dim=1) # [B, encoder_output_dim * 3]
        z = self.fusion(fused_features) # [B, fusion_dim]

        # Multi-task heads
        pfi_pred = self.pfi_head(z).squeeze(-1) # [B]
        tot_logits = self.tot_head(z).squeeze(-1) # [B]
        acl_logits = self.acl_head(z).squeeze(-1) # [B]

        return {
            'pfi_pred': pfi_pred,
            'tot_logits': tot_logits,
            'acl_logits': acl_logits,
            'tot_prob': torch.sigmoid(tot_logits),
            'acl_prob': torch.sigmoid(acl_logits),
            'fusion_embedding': z
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
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

        total_loss = (
            self.loss_weights.get('pfi', 0.0) * losses['pfi_loss'] +
            self.loss_weights.get('tot', 0.0) * losses['tot_loss'] +
            self.loss_weights.get('acl', 0.0) * losses['acl_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses

    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, any]:
        """Get model architecture summary"""
        total_params = self.count_parameters()
        
        return {
            'model_name': 'Tiny-Deep Baseline',
            'total_parameters': total_params,
            'eeg_encoder': '2× 1D-CNN (k=5, 32→64) + GAP',
            'ecg_encoder': '2-layer GRU (64)',
            'gsr_encoder': '1-layer GRU (32)',
            'fusion': 'MLP(256)',
            'output_heads': 'PFI (regression), ToT (binary), ACL (binary)',
            'loss_weights': f"{self.loss_weights['pfi']}·MSE(PFI) + {self.loss_weights['tot']}·BCE(ToT) + {self.loss_weights['acl']}·BCE(ACL)"
        }
