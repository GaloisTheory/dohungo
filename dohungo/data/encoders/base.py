"""Base encoder class for dohungo.

Defines the interface for converting Go board positions to neural network input tensors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class Encoder(ABC):
    """Base class for board position encoders."""
    
    @property
    @abstractmethod
    def num_planes(self) -> int:
        """Number of feature planes this encoder produces."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this encoder."""
        pass
    
    @abstractmethod
    def encode(self, board: np.ndarray, to_play: int) -> torch.Tensor:
        """Encode a board position into feature planes.
        
        Args:
            board: 19x19 numpy array with 1=black, -1=white, 0=empty.
            to_play: Player to move (1=black, -1=white).
            
        Returns:
            Tensor of shape (num_planes, 19, 19) with encoded features.
        """
        pass
    
    def encode_batch(self, boards: list[np.ndarray], to_play_list: list[int]) -> torch.Tensor:
        """Encode a batch of board positions.
        
        Args:
            boards: List of board arrays.
            to_play_list: List of players to move.
            
        Returns:
            Tensor of shape (batch_size, num_planes, 19, 19).
        """
        batch_size = len(boards)
        encoded_batch = torch.zeros(batch_size, self.num_planes, 19, 19)
        
        for i, (board, to_play) in enumerate(zip(boards, to_play_list)):
            encoded_batch[i] = self.encode(board, to_play)
        
        return encoded_batch
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(planes={self.num_planes})" 