"""Simple 11-plane encoder for dohungo.

Encodes Go positions using 11 planes:
- Black stones with 1 liberty (1 plane)
- Black stones with 2 liberties (1 plane)  
- Black stones with 3 liberties (1 plane)
- Black stones with 4+ liberties (1 plane)
- White stones with 1 liberty (1 plane)
- White stones with 2 liberties (1 plane)
- White stones with 3 liberties (1 plane) 
- White stones with 4+ liberties (1 plane)
- Empty points (1 plane)
- Side to play (1 plane)
- Ko position (1 plane)
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Encoder
from ..sgf_reader import count_liberties


class SimpleEncoder(Encoder):
    """Encoder using 11 feature planes with detailed liberty information."""
    
    @property
    def num_planes(self) -> int:
        return 11
    
    @property
    def name(self) -> str:
        return "Simple"
    
    def encode(self, board: np.ndarray, to_play: int) -> torch.Tensor:
        """Encode board position using 11 planes.
        
        Args:
            board: 19x19 board array.
            to_play: Player to move (1=black, -1=white).
            
        Returns:
            Tensor of shape (11, 19, 19).
        """
        size = board.shape[0]
        planes = torch.zeros(11, size, size, dtype=torch.float32)
        
        # Calculate liberty counts for all positions
        liberty_counts = np.zeros_like(board, dtype=np.int32)
        for row in range(size):
            for col in range(size):
                if board[row, col] != 0:
                    liberty_counts[row, col] = count_liberties(board, row, col)
        
        # Planes 0-3: Black stones by liberty count
        for liberty_level in range(1, 5):  # 1, 2, 3, 4+
            if liberty_level == 4:
                # 4+ liberties
                mask = (board == 1) & (liberty_counts >= 4)
            else:
                # Exactly this many liberties
                mask = (board == 1) & (liberty_counts == liberty_level)
            planes[liberty_level - 1] = torch.from_numpy(mask.astype(np.float32))
        
        # Planes 4-7: White stones by liberty count
        for liberty_level in range(1, 5):  # 1, 2, 3, 4+
            if liberty_level == 4:
                # 4+ liberties
                mask = (board == -1) & (liberty_counts >= 4)
            else:
                # Exactly this many liberties
                mask = (board == -1) & (liberty_counts == liberty_level)
            planes[liberty_level + 3] = torch.from_numpy(mask.astype(np.float32))
        
        # Plane 8: Empty points
        planes[8] = torch.from_numpy((board == 0).astype(np.float32))
        
        # Plane 9: Side to play (1 for black, 0 for white)
        side_to_play = 1.0 if to_play == 1 else 0.0
        planes[9] = torch.full((size, size), side_to_play, dtype=torch.float32)
        
        # Plane 10: Ko position (simplified - just zeros for now)
        # TODO: Implement proper ko detection
        ko_plane = torch.zeros(size, size, dtype=torch.float32)
        planes[10] = ko_plane
        
        return planes 