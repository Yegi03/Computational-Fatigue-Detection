"""
PFI (Physiology Fatigue Index) with literature-based weights
Implements the exact formula from Table 1 specifications
"""

import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, List, Tuple, Optional

class PFILiterature:
    """PFI computation with literature-based weights and ridge tuning"""
    
    def __init__(self, initial_weights: List[float] = None):
        """
        Initialize PFI with literature-based weights
        
        Args:
            initial_weights: Initial weights [w1, w2, w3, w4, w5, w6, w7, w8, w9]
        """
        if initial_weights is None:
            # Literature-based initial weights from Table 1
            self.initial_weights = [0.25, 0.15, 0.15, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02]
        else:
            self.initial_weights = initial_weights
        
        self.tuned_weights = None
        self.feature_names = [
            'alpha_beta_ratio',      # w1: α/β
            'alpha_theta_beta_ratio', # w2: (α+θ)/β
            'lzc',                   # w3: Lempel-Ziv Complexity
            'lf_hf_ratio',           # w4: LF/HF
            'rmssd',                 # w5: RMSSD
            'sdnn',                  # w6: SDNN
            'scr_freq',              # w7: SCR frequency
            'scr_amp_mean',          # w8: SCR amplitude
            'scl_mean'               # w9: SCL mean
        ]
        
        self.ridge_model = Ridge(alpha=1.0)
    
    def compute_pfi(self, features: Dict[str, float], weights: List[float] = None) -> float:
        """
        Compute PFI using the literature formula
        
        Formula: PFI = w₁·(α/β) + w₂·((α+θ)/β) - w₃·LZC + w₄·(LF/HF) - w₅·RMSSD - w₆·SDNN + w₇·SCR_freq + w₈·SCR_amp + w₉·SCL
        
        Args:
            features: Dictionary with feature values
            weights: Weights to use (if None, use tuned or initial weights)
            
        Returns:
            PFI score (higher = more fatigued)
        """
        if weights is None:
            weights = self.tuned_weights if self.tuned_weights is not None else self.initial_weights
        
        # Extract features in the correct order
        feature_values = []
        for feature_name in self.feature_names:
            # Map feature names to actual feature keys
            if feature_name == 'alpha_beta_ratio':
                key = 'eeg_alpha_beta_ratio'
            elif feature_name == 'alpha_theta_beta_ratio':
                key = 'eeg_alpha_theta_beta_ratio'
            elif feature_name == 'lzc':
                key = 'eeg_lzc'
            elif feature_name == 'lf_hf_ratio':
                key = 'ecg_lf_hf_ratio'
            elif feature_name == 'rmssd':
                key = 'ecg_rmssd'
            elif feature_name == 'sdnn':
                key = 'ecg_sdnn'
            elif feature_name == 'scr_freq':
                key = 'gsr_scr_freq'
            elif feature_name == 'scr_amp_mean':
                key = 'gsr_scr_amp_mean'
            elif feature_name == 'scl_mean':
                key = 'gsr_scl_mean'
            else:
                key = feature_name
            
            value = features.get(key, np.nan)
            feature_values.append(value)
        
        # Check for NaN values
        if np.any(np.isnan(feature_values)):
            return np.nan
        
        # Apply formula with signs
        signs = [1, 1, -1, 1, -1, -1, 1, 1, 1]  # Signs from formula
        
        pfi = 0.0
        for i, (weight, value, sign) in enumerate(zip(weights, feature_values, signs)):
            pfi += sign * weight * value
        
        return pfi
    
    def tune_weights(self, features_matrix: np.ndarray, targets: np.ndarray) -> List[float]:
        """
        Tune PFI weights using ridge regression on proxies (train subjects only)
        
        Args:
            features_matrix: Feature matrix [N, 9] with features in order
            targets: Target values [N] (PFI targets or proxies)
            
        Returns:
            Tuned weights
        """
        # Remove NaN values
        valid_mask = ~np.isnan(features_matrix).any(axis=1) & ~np.isnan(targets)
        X = features_matrix[valid_mask]
        y = targets[valid_mask]
        
        if len(X) < 10:  # Need sufficient data
            return self.initial_weights
        
        # Fit ridge regression
        self.ridge_model.fit(X, y)
        
        # Get tuned weights
        self.tuned_weights = self.ridge_model.coef_.tolist()
        
        return self.tuned_weights
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from tuned weights
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.tuned_weights is None:
            return {name: weight for name, weight in zip(self.feature_names, self.initial_weights)}
        
        return {name: abs(weight) for name, weight in zip(self.feature_names, self.tuned_weights)}
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = []
        
        for feature_name in self.feature_names:
            # Map to actual feature keys
            if feature_name == 'alpha_beta_ratio':
                key = 'eeg_alpha_beta_ratio'
            elif feature_name == 'alpha_theta_beta_ratio':
                key = 'eeg_alpha_theta_beta_ratio'
            elif feature_name == 'lzc':
                key = 'eeg_lzc'
            elif feature_name == 'lf_hf_ratio':
                key = 'ecg_lf_hf_ratio'
            elif feature_name == 'rmssd':
                key = 'ecg_rmssd'
            elif feature_name == 'sdnn':
                key = 'ecg_sdnn'
            elif feature_name == 'scr_freq':
                key = 'gsr_scr_freq'
            elif feature_name == 'scr_amp_mean':
                key = 'gsr_scr_amp_mean'
            elif feature_name == 'scl_mean':
                key = 'gsr_scl_mean'
            else:
                key = feature_name
            
            if key not in features or np.isnan(features[key]):
                missing.append(feature_name)
        
        return len(missing) == 0, missing
    
    def get_formula_string(self) -> str:
        """Get the PFI formula as a string"""
        return r"PFI = w₁·(α/β) + w₂·((α+θ)/β) - w₃·LZC + w₄·(LF/HF) - w₅·RMSSD - w₆·SDNN + w₇·SCR_freq + w₈·SCR_amp + w₉·SCL"
    
    def get_weights_summary(self) -> Dict[str, Dict[str, float]]:
        """Get weights summary for reporting"""
        weights_to_use = self.tuned_weights if self.tuned_weights is not None else self.initial_weights
        
        summary = {}
        for i, (name, weight) in enumerate(zip(self.feature_names, weights_to_use)):
            summary[name] = {
                'weight': weight,
                'sign': 'positive' if i in [0, 1, 3, 6, 7, 8] else 'negative',
                'description': self._get_feature_description(name)
            }
        
        return summary
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of feature"""
        descriptions = {
            'alpha_beta_ratio': 'EEG α/β power ratio',
            'alpha_theta_beta_ratio': 'EEG (α+θ)/β power ratio',
            'lzc': 'EEG Lempel-Ziv Complexity',
            'lf_hf_ratio': 'ECG LF/HF ratio',
            'rmssd': 'ECG RMSSD (ms)',
            'sdnn': 'ECG SDNN (ms)',
            'scr_freq': 'GSR SCR frequency (peaks/min)',
            'scr_amp_mean': 'GSR SCR amplitude (µS)',
            'scl_mean': 'GSR SCL mean (µS)'
        }
        return descriptions.get(feature_name, feature_name)
