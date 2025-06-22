"""Tests for dohungo encoders."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dohungo.data.encoders import SevenPlaneEncoder, SimpleEncoder


def test_seven_plane_encoder():
    """Test SevenPlaneEncoder functionality."""
    encoder = SevenPlaneEncoder()
    
    # Test properties
    assert encoder.num_planes == 7
    assert encoder.name == "SevenPlane"
    
    # Create a simple board state
    board = np.zeros((19, 19), dtype=np.int8)
    board[3, 3] = 1  # Black stone
    board[4, 4] = -1  # White stone
    to_play = 1  # Black to play
    
    # Encode the position
    encoded = encoder.encode(board, to_play)
    
    # Check output shape
    assert encoded.shape == (7, 19, 19)
    assert isinstance(encoded, torch.Tensor)
    
    # Check that black stone plane has the stone
    assert encoded[0, 3, 3] == 1.0
    assert encoded[0, 4, 4] == 0.0  # No black stone here
    
    # Check that white stone plane has the stone
    assert encoded[1, 4, 4] == 1.0
    assert encoded[1, 3, 3] == 0.0  # No white stone here
    
    # Check empty points plane
    assert encoded[2, 0, 0] == 1.0  # Empty point
    assert encoded[2, 3, 3] == 0.0  # Not empty
    
    # Check side to play plane
    assert encoded[6, 0, 0] == 1.0  # Black to play


def test_simple_encoder():
    """Test SimpleEncoder functionality."""
    encoder = SimpleEncoder()
    
    # Test properties
    assert encoder.num_planes == 11
    assert encoder.name == "Simple"
    
    # Create a simple board state
    board = np.zeros((19, 19), dtype=np.int8)
    board[3, 3] = 1  # Black stone
    board[4, 4] = -1  # White stone
    to_play = -1  # White to play
    
    # Encode the position
    encoded = encoder.encode(board, to_play)
    
    # Check output shape
    assert encoded.shape == (11, 19, 19)
    assert isinstance(encoded, torch.Tensor)
    
    # Check empty points plane
    assert encoded[8, 0, 0] == 1.0  # Empty point
    assert encoded[8, 3, 3] == 0.0  # Not empty
    
    # Check side to play plane
    assert encoded[9, 0, 0] == 0.0  # White to play (0 for white)


def test_encoder_batch_encoding():
    """Test batch encoding functionality."""
    encoder = SevenPlaneEncoder()
    
    # Create two different board states
    board1 = np.zeros((19, 19), dtype=np.int8)
    board1[3, 3] = 1
    
    board2 = np.zeros((19, 19), dtype=np.int8)
    board2[4, 4] = -1
    
    boards = [board1, board2]
    to_play_list = [1, -1]
    
    # Batch encode
    batch_encoded = encoder.encode_batch(boards, to_play_list)
    
    # Check output shape
    assert batch_encoded.shape == (2, 7, 19, 19)
    
    # Check individual encodings match
    individual1 = encoder.encode(board1, 1)
    individual2 = encoder.encode(board2, -1)
    
    assert torch.allclose(batch_encoded[0], individual1)
    assert torch.allclose(batch_encoded[1], individual2)


def test_encoder_edge_cases():
    """Test edge cases for encoders."""
    encoder = SevenPlaneEncoder()
    
    # Empty board
    empty_board = np.zeros((19, 19), dtype=np.int8)
    encoded = encoder.encode(empty_board, 1)
    
    # All positions should be empty
    assert torch.all(encoded[2] == 1.0)  # Empty plane
    assert torch.all(encoded[0] == 0.0)  # No black stones
    assert torch.all(encoded[1] == 0.0)  # No white stones
    
    # Full board (alternating pattern)
    full_board = np.ones((19, 19), dtype=np.int8)
    full_board[::2, ::2] = 1   # Black on even positions
    full_board[1::2, 1::2] = 1  # Black on odd positions  
    full_board[::2, 1::2] = -1  # White on mixed positions
    full_board[1::2, ::2] = -1  # White on mixed positions
    
    encoded = encoder.encode(full_board, 1)
    
    # No empty positions
    assert torch.all(encoded[2] == 0.0)  # Empty plane should be all zeros


if __name__ == "__main__":
    # Run tests directly
    test_seven_plane_encoder()
    test_simple_encoder() 
    test_encoder_batch_encoding()
    test_encoder_edge_cases()
    print("All encoder tests passed!") 