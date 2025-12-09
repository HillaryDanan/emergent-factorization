"""
Neural network architectures for factorization experiments.

Design principle: Keep architectures simple to isolate the effect
of the manipulation (bottleneck size, multi-task pressure).

Complex architectures introduce confounds. Simplicity aids interpretation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class BottleneckAutoencoder(nn.Module):
    """
    Simple autoencoder with variable bottleneck size.
    
    Architecture:
        Input -> Hidden layers -> Bottleneck -> Hidden layers -> Output
    
    The bottleneck is the key manipulation point.
    
    Hypothesis (APH Section 9.2):
    - Small bottleneck creates factorization PRESSURE
    - If data has factorial structure and bottleneck is sized appropriately,
      the network should discover factorized representations
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        
        # Select activation
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        
        # Build encoder
        encoder_layers = []
        current_dim = input_dim
        
        for _ in range(n_hidden_layers):
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                act_fn
            ])
            current_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(current_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        current_dim = bottleneck_dim
        
        for _ in range(n_hidden_layers):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                act_fn
            ])
            current_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get bottleneck representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from bottleneck."""
        return self.decoder(z)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: Output logits for reconstruction
            bottleneck: Bottleneck representation (for analysis)
        """
        z = self.encode(x)
        logits = self.decode(z)
        return logits, z


class MultiTaskEncoder(nn.Module):
    """
    Encoder with multiple task heads sharing a bottleneck.
    
    Architecture:
        Input -> Hidden -> Bottleneck -> Task 1 head (classify factor 1)
                                      -> Task 2 head (classify factor 2)
                                      -> Task 3 head (identify combination)
    
    Hypothesis (APH Section 9.2, Condition 4 - Multi-task Learning):
    - A holistic encoder CANNOT solve Tasks 1 and 2 (need to ignore one factor)
    - Multi-task pressure should FORCE factorized representations
    - Compositional generalization on Task 3 should emerge as a SIDE EFFECT
    
    This is a stronger test than bottleneck pressure alone because:
    - Bottleneck pressure: factorization is the EFFICIENT solution
    - Multi-task pressure: factorization is the ONLY solution
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        n_factor1: int,
        n_factor2: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_factor1 = n_factor1
        self.n_factor2 = n_factor2
        
        # Shared encoder to bottleneck
        encoder_layers = []
        current_dim = input_dim
        
        for _ in range(n_hidden_layers):
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(current_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Task 1: Classify factor 1 (e.g., color)
        self.head_factor1 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor1)
        )
        
        # Task 2: Classify factor 2 (e.g., shape)
        self.head_factor2 = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor2)
        )
        
        # Task 3: Identify full combination
        self.head_combination = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factor1 * n_factor2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get bottleneck representation."""
        return self.encoder(x)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for all tasks.
        
        Returns:
            logits_f1: Logits for factor 1 classification
            logits_f2: Logits for factor 2 classification
            logits_comb: Logits for combination identification
            bottleneck: Bottleneck representation
        """
        z = self.encode(x)
        
        logits_f1 = self.head_factor1(z)
        logits_f2 = self.head_factor2(z)
        logits_comb = self.head_combination(z)
        
        return logits_f1, logits_f2, logits_comb, z