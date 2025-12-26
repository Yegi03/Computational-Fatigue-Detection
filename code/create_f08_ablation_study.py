#!/usr/bin/env python3
"""
Create F08: Ablation Study Results
Compare all 4 variants: Full FT, LoRA-only, MoE-only, LoRA+MoE
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_f08_ablation_study(output_dir: Path):
    """Create F08: Ablation Study - All Variants Comparison"""
    # Data from ablation study results
    methods = ['Full\nFine-tuning', 'LoRA-only\n(r=16)', 'MoE-only\n(K=3)', 'LoRA+MoE*\n(r=16, K=3)']
    
    # Performance metrics
    pfi_ccc = [0.587, 0.620, 0.635, 0.680]
    pfi_mae = [0.118, 0.112, 0.105, 0.095]
    tot_f1 = [0.523, 0.571, 0.560, 0.585]
    ece = [0.040, 0.036, 0.033, 0.028]
    
    # Parameter counts
    params = [2200000, 50000, 282755, 332755]
    param_percent = [100.0, 2.3, 12.9, 15.1]
    
    # Colors - pastel professional
    colors = ['#FF6B6B', '#4ECDC4', '#FFD700', '#90EE90']  # Red, Teal, Gold, Light Green
    
    # Create figure with 1x2 subplots (2 panels instead of 4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Set style - consistent with other figures
    plt.rcParams.update({'font.size': 12})
    
    # Left panel: PFI CCC (main regression metric)
    bars1 = ax1.bar(methods, pfi_ccc, color=colors, alpha=0.85, edgecolor='gray', linewidth=1.2)
    ax1.set_ylabel('PFI CCC (higher is better)', fontsize=12, fontweight='bold')
    ax1.set_title('PFI Concordance Correlation Coefficient', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylim(0.5, 0.75)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=0, labelsize=11)
    
    # Add value labels
    for bar, value in zip(bars1, pfi_ccc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight LoRA+MoE
    bars1[3].set_edgecolor('darkgreen')
    bars1[3].set_linewidth(3)
    
    # Right panel: PFI MAE (regression metric - lower is better)
    bars2 = ax2.bar(methods, pfi_mae, color=colors, alpha=0.85, edgecolor='gray', linewidth=1.2)
    ax2.set_ylabel('PFI MAE (lower is better)', fontsize=12, fontweight='bold')
    ax2.set_title('PFI Mean Absolute Error', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylim(0.08, 0.13)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=0, labelsize=11)
    
    # Add value labels
    for bar, value in zip(bars2, pfi_mae):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight LoRA+MoE (lowest MAE = best)
    bars2[3].set_edgecolor('darkgreen')
    bars2[3].set_linewidth(3)
    
    # Add overall title - focused on PFI regression since both panels are PFI metrics
    fig.suptitle('Ablation Study: PFI Regression Performance Across Variants', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.15)  # Reduced wspace from 0.25 to 0.15 to bring columns closer
    plt.savefig(output_dir / "F08_ablation_study.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "F08_ablation_study.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ F08_ablation_study.pdf/.png")

if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_f08_ablation_study(output_dir)

