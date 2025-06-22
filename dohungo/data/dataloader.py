"""DataLoader for streaming SGF data in dohungo."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from .downloader import iter_sgf_files
from .encoders.base import Encoder
from .sgf_reader import Position, read_sgf_file


class SGFDataset(IterableDataset):
    """Iterable dataset for streaming SGF positions."""
    
    def __init__(
        self,
        data_dir: Path = Path("data/raw"),
        encoder: Optional[Encoder] = None,
        board_size: int = 19,
        max_positions_per_game: int = 200,
        skip_first_moves: int = 10,
    ):
        """Initialize SGF dataset.
        
        Args:
            data_dir: Directory containing SGF files.
            encoder: Board position encoder.
            board_size: Board size to filter.
            max_positions_per_game: Maximum positions to extract per game.
            skip_first_moves: Skip opening moves (often handicap).
        """
        self.data_dir = data_dir
        self.encoder = encoder
        self.board_size = board_size
        self.max_positions_per_game = max_positions_per_game
        self.skip_first_moves = skip_first_moves
    
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over (encoded_position, move_target) pairs.
        
        Yields:
            Tuple of (encoded_board, move_target) where:
            - encoded_board: tensor of shape (num_planes, 19, 19)
            - move_target: tensor of shape (361,) - one-hot encoded move
        """
        for sgf_file in iter_sgf_files(self.data_dir, self.board_size):
            try:
                positions_yielded = 0
                for position in read_sgf_file(sgf_file, self.board_size):
                    # Skip early moves and passes
                    if (position.move_number < self.skip_first_moves or 
                        position.next_move is None or
                        positions_yielded >= self.max_positions_per_game):
                        continue
                    
                    # Encode the board position
                    if self.encoder is not None:
                        encoded_board = self.encoder.encode(position.board, position.to_play)
                    else:
                        # Default: just use the board as a single plane
                        encoded_board = torch.from_numpy(position.board).unsqueeze(0).float()
                    
                    # Create move target (one-hot encoding)
                    move_target = torch.zeros(self.board_size * self.board_size)
                    if position.next_move is not None:
                        row, col = position.next_move
                        move_idx = row * self.board_size + col
                        move_target[move_idx] = 1.0
                    
                    yield encoded_board, move_target
                    positions_yielded += 1
                    
            except (ValueError, IOError) as e:
                # Skip corrupted files
                print(f"Skipping {sgf_file}: {e}")
                continue


def create_dataloader(
    data_dir: Path = Path("data/raw"),
    encoder: Optional[Encoder] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    board_size: int = 19,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for SGF data.
    
    Args:
        data_dir: Directory containing SGF files.
        encoder: Board position encoder.
        batch_size: Batch size for training.
        num_workers: Number of worker processes.
        board_size: Board size.
        
    Returns:
        PyTorch DataLoader instance.
    """
    dataset = SGFDataset(
        data_dir=data_dir,
        encoder=encoder,
        board_size=board_size,
    )
    
    # For deterministic behavior
    def worker_init_fn(worker_id: int) -> None:
        import numpy as np
        import random
        np.random.seed(2025 + worker_id)
        random.seed(2025 + worker_id)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    ) 