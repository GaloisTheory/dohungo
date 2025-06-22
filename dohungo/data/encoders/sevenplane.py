"""Seven-plane encoder for dohungo.

Encodes Go positions using 7 planes:
- Black stones (1 plane)
- White stones (1 plane) 
- Empty points (1 plane)
- Black liberties (1 plane)
- White liberties (1 plane)
- Ko position (1 plane)
- Side to play (1 plane)
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Encoder
from ..sgf_reader import count_liberties


class SevenPlaneEncoder(Encoder):
    """Encoder using 7 feature planes."""
    
    @property
    def num_planes(self) -> int:
        return 7
    
    @property
    def name(self) -> str:
        return "SevenPlane"
    
    def encode(self, board: np.ndarray, to_play: int) -> torch.Tensor:
        """Encode board position using 7 planes.
        
        Args:
            board: 19x19 board array.
            to_play: Player to move (1=black, -1=white).
            
        Returns:
            Tensor of shape (7, 19, 19).
        """
        size = board.shape[0]
        planes = torch.zeros(7, size, size, dtype=torch.float32)
        
        # Plane 0: Black stones
        planes[0] = torch.from_numpy((board == 1).astype(np.float32))
        
        # Plane 1: White stones  
        planes[1] = torch.from_numpy((board == -1).astype(np.float32))
        
        # Plane 2: Empty points
        planes[2] = torch.from_numpy((board == 0).astype(np.float32))
        
        # Plane 3: Black liberties (normalized by max possible = 4)
        black_liberties = np.zeros_like(board, dtype=np.float32)
        for row in range(size):
            for col in range(size):
                if board[row, col] == 1:  # Black stone
                    liberties = count_liberties(board, row, col)
                    black_liberties[row, col] = min(liberties, 4) / 4.0
        planes[3] = torch.from_numpy(black_liberties)
        
        # Plane 4: White liberties (normalized by max possible = 4)
        white_liberties = np.zeros_like(board, dtype=np.float32)
        for row in range(size):
            for col in range(size):
                if board[row, col] == -1:  # White stone
                    liberties = count_liberties(board, row, col)
                    white_liberties[row, col] = min(liberties, 4) / 4.0
        planes[4] = torch.from_numpy(white_liberties)
        
        # Plane 5: Ko position (simplified - just mark recently captured spots)
        # TODO: Implement proper ko detection
        ko_plane = torch.zeros(size, size, dtype=torch.float32)
        planes[5] = ko_plane
        
        # Plane 6: Side to play (1 for black, 0 for white)
        side_to_play = 1.0 if to_play == 1 else 0.0
        planes[6] = torch.full((size, size), side_to_play, dtype=torch.float32)
        
        return planes 