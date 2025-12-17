"""Policy heads for placement action selection.

Implements three policy head architectures for the 4×16×8 placement grid:
- ShiftAndScoreHead: Parallel local scorer with partner-shifted features
- DenseConvHead: Direct heatmap generation with FiLM conditioning
- FactorizedHead: Hierarchical anchor→orientation selection

All heads share the same interface and produce [B, 4, 16, 8] logit maps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# CoordConv helper
class CoordConv2d(nn.Module):
    """2D convolution with coordinate channels appended."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        **kwargs
    ):
        super().__init__()
        # +2 for (i, j) coordinate channels
        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device
        
        # Create coordinate grids in [0, 1]
        i_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        j_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Concatenate
        x_with_coords = torch.cat([x, i_coords, j_coords], dim=1)
        return self.conv(x_with_coords)


class DrMarioBoardEncoder(nn.Module):
    """Tiny CNN encoder for 16×8 board state with coordinate awareness.
    
    Takes multi-channel board state and produces spatial features [B, 64, 16, 8].
    """
    
    def __init__(self, in_ch: int = 12):
        super().__init__()
        # Small, efficient encoder
        self.conv1 = CoordConv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """Encode board state.
        
        Args:
            x: Board tensor [B, C, 16, 8] with C channels
            
        Returns:
            Feature map [B, 64, 16, 8]
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return x


class UnorderedPillEmbedding(nn.Module):
    """Deep-Sets style embedding for unordered next-pill colors.
    
    Ensures that (c1, c2) and (c2, c1) produce identical embeddings
    via symmetric sum and element-wise product pooling.
    """
    
    def __init__(self, num_colors: int = 3, embedding_dim: int = 16, output_dim: int = 32):
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, embedding_dim)
        # Map sum and product to output
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, colors):
        """Embed unordered color pair.
        
        Args:
            colors: Tensor [B, 2] with color indices in {0, 1, 2}
            
        Returns:
            Embedding [B, output_dim]
        """
        # colors: [B, 2]
        c1 = self.color_embed(colors[:, 0])  # [B, D]
        c2 = self.color_embed(colors[:, 1])  # [B, D]
        
        # Symmetric pooling
        p_sum = c1 + c2
        p_mul = c1 * c2
        
        # Concatenate and map
        combined = torch.cat([p_sum, p_mul], dim=-1)  # [B, 2D]
        return self.mlp(combined)


class ShiftAndScoreHead(nn.Module):
    """Shift-and-Score placement head.
    
    For each (o, i, j):
    - Compute features at anchor (i, j)
    - Compute features at partner cell shifted by orientation o
    - Concatenate with pill embedding
    - Score with tiny 1×1 MLP
    
    All operations are parallel (no loops over actions).
    """
    
    def __init__(self, P: int = 32):
        super().__init__()
        self.P = P
        
        # Scorer: (64 anchor + 64 partner + P pill) → 1
        # Use 1×1 convs for efficiency
        self.scorer = nn.Sequential(
            nn.Conv2d(64 + 64 + P, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 4, kernel_size=1),  # 4 orientation channels
        )
        
    def forward(self, F, p_embed, mask):
        """Compute placement logits.
        
        Args:
            F: Spatial features [B, 64, 16, 8]
            p_embed: Pill embedding [B, P]
            mask: Feasibility mask [B, 4, 16, 8]
            
        Returns:
            Tuple (logits_map [B, 4, 16, 8], value [B, 1])
        """
        B, C, H, W = F.shape
        
        # Precompute partner-shifted features for each orientation
        # o=0: horizontal+, partner at (i, j+1)
        # o=1: vertical+, partner at (i+1, j)
        # o=2: horizontal−, partner at (i, j-1)
        # o=3: vertical−, partner at (i-1, j)
        
        G = torch.zeros(B, 4, C, H, W, device=F.device, dtype=F.dtype)
        
        # o=0: shift left (partner is to the right)
        G[:, 0, :, :, :-1] = F[:, :, :, 1:]
        
        # o=1: shift up (partner is below)
        G[:, 1, :, :-1, :] = F[:, :, 1:, :]
        
        # o=2: shift right (partner is to the left)
        G[:, 2, :, :, 1:] = F[:, :, :, :-1]
        
        # o=3: shift down (partner is above)
        G[:, 3, :, 1:, :] = F[:, :, :-1, :]
        
        # Broadcast anchor features and pill embedding spatially
        F4 = F.unsqueeze(1).expand(B, 4, C, H, W)  # [B, 4, 64, 16, 8]
        P_spatial = p_embed.view(B, self.P, 1, 1).expand(B, self.P, H, W)  # [B, P, 16, 8]
        P4 = P_spatial.unsqueeze(1).expand(B, 4, self.P, H, W)  # [B, 4, P, 16, 8]
        
        # Concatenate anchor, partner, pill
        X = torch.cat([F4, G, P4], dim=2)  # [B, 4, 64+64+P, 16, 8]
        
        # Reshape for conv: [B*4, C', 16, 8]
        X_flat = X.reshape(B * 4, 64 + 64 + self.P, H, W)
        
        # Score: [B*4, 4, 16, 8] - wait, this produces 4 channels per orientation?
        # We want 1 channel per orientation. Fix:
        # Actually the scorer should output 1 channel, then we reshape.
        # Let me reconsider the architecture.
        
        # Actually, let's make scorer output 1 channel:
        # Then reshape [B*4, 1, 16, 8] → [B, 4, 16, 8]
        
        # ERROR in my design above. Let me fix:
        pass  # Will be corrected in replacement below


class ShiftAndScoreHeadCorrected(nn.Module):
    """Shift-and-Score placement head (corrected version)."""
    
    def __init__(self, P: int = 32):
        super().__init__()
        self.P = P
        
        # Scorer: (64 anchor + 64 partner + P pill) → 1 score per location
        self.scorer = nn.Sequential(
            nn.Conv2d(64 + 64 + P, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 1, kernel_size=1),  # 1 score channel
        )
        
    def forward(self, F, p_embed, mask):
        """Compute placement logits.
        
        Args:
            F: Spatial features [B, 64, 16, 8]
            p_embed: Pill embedding [B, P]
            mask: Feasibility mask [B, 4, 16, 8]
            
        Returns:
            logits_map [B, 4, 16, 8]
        """
        B, C, H, W = F.shape
        
        # Partner shifts
        G = torch.zeros(B, 4, C, H, W, device=F.device, dtype=F.dtype)
        G[:, 0, :, :, :-1] = F[:, :, :, 1:]   # o=0: H+
        G[:, 1, :, :-1, :] = F[:, :, 1:, :]   # o=1: V+
        G[:, 2, :, :, 1:] = F[:, :, :, :-1]   # o=2: H−
        G[:, 3, :, 1:, :] = F[:, :, :-1, :]   # o=3: V−
        
        # Broadcast
        F4 = F.unsqueeze(1).expand(B, 4, C, H, W)
        P_spatial = p_embed.view(B, self.P, 1, 1).expand(B, self.P, H, W)
        P4 = P_spatial.unsqueeze(1).expand(B, 4, self.P, H, W)
        
        # Concatenate
        X = torch.cat([F4, G, P4], dim=2)  # [B, 4, 128+P, 16, 8]
        
        # Reshape and score
        X_flat = X.reshape(B * 4, 64 + 64 + self.P, H, W)
        scores_flat = self.scorer(X_flat)  # [B*4, 1, 16, 8]
        
        # Reshape back
        logits_map = scores_flat.reshape(B, 4, H, W)
        
        return logits_map


class DenseConvHead(nn.Module):
    """Dense convolutional heatmap head with FiLM conditioning."""
    
    def __init__(self, P: int = 32):
        super().__init__()
        self.P = P
        
        # FiLM: generate scale and shift for 64 channels from pill embedding
        self.film_scale = nn.Linear(P, 64)
        self.film_shift = nn.Linear(P, 64)
        
        # Neck: 64 → 4 orientation heatmaps
        self.neck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
        )
        
    def forward(self, F, p_embed, mask):
        """Compute placement logits.
        
        Args:
            F: Spatial features [B, 64, 16, 8]
            p_embed: Pill embedding [B, P]
            mask: Feasibility mask [B, 4, 16, 8]
            
        Returns:
            logits_map [B, 4, 16, 8]
        """
        B, C, H, W = F.shape
        
        # FiLM modulation
        scale = self.film_scale(p_embed).view(B, C, 1, 1)  # [B, 64, 1, 1]
        shift = self.film_shift(p_embed).view(B, C, 1, 1)
        
        F_modulated = F * (1.0 + scale) + shift
        
        # Generate heatmaps
        logits_map = self.neck(F_modulated)  # [B, 4, 16, 8]
        
        return logits_map


class FactorizedHead(nn.Module):
    """Factorized placement head: anchor→orientation.
    
    First samples anchor cell (i, j), then orientation o | (i, j).
    """
    
    def __init__(self, P: int = 32):
        super().__init__()
        self.P = P
        
        # Anchor heatmap: 64 → 1
        self.anchor_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # Orientation scorer: (64 anchor + GAP(64) + P pill) → 4
        self.orient_mlp = nn.Sequential(
            nn.Linear(64 + 64 + P, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )
        
    def forward(self, F, p_embed, mask):
        """Compute placement logits (factorized).
        
        Note: For training, we still produce a joint [B, 4, 16, 8] logit map
        by combining anchor and orientation logits. At inference, this can be
        sampled hierarchically.
        
        Args:
            F: Spatial features [B, 64, 16, 8]
            p_embed: Pill embedding [B, P]
            mask: Feasibility mask [B, 4, 16, 8]
            
        Returns:
            logits_map [B, 4, 16, 8]
        """
        B, C, H, W = F.shape
        
        # Anchor logits: [B, 1, 16, 8]
        anchor_logits = self.anchor_head(F)
        
        # Global context
        F_gap = F.mean(dim=(2, 3))  # [B, 64]
        
        # For each spatial location, compute orientation scores
        # Reshape F: [B, 64, 16, 8] → [B, 16, 8, 64]
        F_spatial = F.permute(0, 2, 3, 1)  # [B, 16, 8, 64]
        
        # Concatenate per-location features with global and pill
        # Broadcast F_gap and p_embed
        F_gap_bc = F_gap.unsqueeze(1).unsqueeze(1).expand(B, H, W, 64)  # [B, 16, 8, 64]
        p_embed_bc = p_embed.unsqueeze(1).unsqueeze(1).expand(B, H, W, self.P)
        
        features_concat = torch.cat([F_spatial, F_gap_bc, p_embed_bc], dim=-1)  # [B, 16, 8, 128+P]
        
        # Reshape and apply MLP
        features_flat = features_concat.reshape(B * H * W, 64 + 64 + self.P)
        orient_logits_flat = self.orient_mlp(features_flat)  # [B*16*8, 4]
        orient_logits = orient_logits_flat.reshape(B, H, W, 4).permute(0, 3, 1, 2)  # [B, 4, 16, 8]
        
        # Combine: broadcast anchor logits across orientations and add
        anchor_bc = anchor_logits.expand(B, 4, H, W)
        logits_map = anchor_bc + orient_logits
        
        return logits_map


# Alias for the corrected version
ShiftAndScoreHead = ShiftAndScoreHeadCorrected


class PlacementPolicyNet(nn.Module):
    """Complete placement policy network.
    
    Combines encoder, pill embedding, policy head, and value head.
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        head_type: str = "dense",
        pill_embed_dim: int = 32,
        num_colors: int = 3,
    ):
        super().__init__()
        
        self.encoder = DrMarioBoardEncoder(in_ch=in_channels)
        self.pill_embedding = UnorderedPillEmbedding(
            num_colors=num_colors,
            embedding_dim=16,
            output_dim=pill_embed_dim,
        )
        
        # Select head
        head_type = head_type.lower()
        if head_type == "shift_score":
            self.head = ShiftAndScoreHead(P=pill_embed_dim)
        elif head_type == "dense":
            self.head = DenseConvHead(P=pill_embed_dim)
        elif head_type == "factorized":
            self.head = FactorizedHead(P=pill_embed_dim)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
            
        # Value head: global average pool → MLP
        self.value_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        
    def forward(self, board, next_pill_colors, mask):
        """Forward pass.
        
        Args:
            board: Board state [B, C, 16, 8]
            next_pill_colors: Color indices [B, 2]
            mask: Feasibility mask [B, 4, 16, 8]
            
        Returns:
            Tuple (logits_map [B, 4, 16, 8], value [B, 1])
        """
        # Encode board
        F = self.encoder(board)  # [B, 64, 16, 8]
        
        # Embed pill colors
        p_embed = self.pill_embedding(next_pill_colors)  # [B, P]
        
        # Generate placement logits
        logits_map = self.head(F, p_embed, mask)  # [B, 4, 16, 8]
        
        # Value estimate
        F_gap = F.mean(dim=(2, 3))  # [B, 64]
        value = self.value_head(F_gap)  # [B, 1]
        
        return logits_map, value


__all__ = [
    "DrMarioBoardEncoder",
    "UnorderedPillEmbedding",
    "ShiftAndScoreHead",
    "DenseConvHead",
    "FactorizedHead",
    "PlacementPolicyNet",
    "CoordConv2d",
]
