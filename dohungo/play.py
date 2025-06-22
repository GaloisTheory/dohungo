"""Interactive play script for dohungo.

Usage:
    python -m dohungo.play --model checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .data.encoders import SevenPlaneEncoder, SimpleEncoder
from .models.cnn_small import CNNSmall


class GoGame:
    """Simple Go game state for interactive play."""
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 = black, -1 = white
        self.move_history = []
        self.passed = False
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid."""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move on the board."""
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        self.current_player = -self.current_player
        self.passed = False
        return True
    
    def pass_turn(self) -> None:
        """Pass the current turn."""
        self.current_player = -self.current_player
        self.passed = True
    
    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get all legal moves."""
        legal_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col):
                    legal_moves.append((row, col))
        return legal_moves
    
    def display_board(self) -> None:
        """Display the current board state."""
        print("   " + " ".join(f"{i:2d}" for i in range(self.board_size)))
        for row in range(self.board_size):
            row_str = f"{row:2d} "
            for col in range(self.board_size):
                if self.board[row, col] == 1:
                    row_str += " ●"  # Black stone
                elif self.board[row, col] == -1:
                    row_str += " ○"  # White stone
                else:
                    row_str += " ·"  # Empty
            print(row_str)
    
    def is_game_over(self) -> bool:
        """Check if the game is over (simplified)."""
        return len(self.get_legal_moves()) == 0


class DohungoPlayer:
    """AI player using the trained dohungo model."""
    
    def __init__(self, model_path: Path, device: torch.device):
        self.device = device
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
        # Create encoder
        encoder_name = config.model.encoder
        if encoder_name == "sevenplane":
            self.encoder = SevenPlaneEncoder()
        elif encoder_name == "simple":
            self.encoder = SimpleEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Create and load model
        self.model = CNNSmall(
            input_planes=self.encoder.num_planes,
            board_size=config.data.board_size,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Model accuracy: {checkpoint['best_accuracy']:.4f}")
    
    def get_move(self, game: GoGame) -> tuple[int, int] | None:
        """Get the AI's move."""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        # Encode the current board position
        encoded_board = self.encoder.encode(game.board, game.current_player)
        encoded_board = encoded_board.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get move probabilities
        with torch.no_grad():
            move_probs = self.model.predict_move(encoded_board)
        
        # Convert to numpy and reshape to board
        move_probs = move_probs.cpu().numpy().reshape(game.board_size, game.board_size)
        
        # Mask illegal moves
        for row in range(game.board_size):
            for col in range(game.board_size):
                if not game.is_valid_move(row, col):
                    move_probs[row, col] = 0.0
        
        # Choose move with highest probability
        if move_probs.sum() == 0:
            # All moves have zero probability, choose randomly
            return random.choice(legal_moves)
        
        # Find the best move
        best_row, best_col = np.unravel_index(np.argmax(move_probs), move_probs.shape)
        return (best_row, best_col)


def play_against_ai(model_path: Path, device: torch.device) -> None:
    """Play an interactive game against the AI."""
    game = GoGame(board_size=19)
    ai_player = DohungoPlayer(model_path, device)
    
    print("Welcome to dohungo! You are playing as Black (●).")
    print("Enter moves as 'row col' (e.g., '3 3'), or 'pass' to pass, or 'quit' to exit.")
    print()
    
    while not game.is_game_over():
        game.display_board()
        
        if game.current_player == 1:  # Human player (black)
            print(f"Your turn (Black ●):")
            move_input = input("Enter move: ").strip().lower()
            
            if move_input == "quit":
                break
            elif move_input == "pass":
                game.pass_turn()
                print("You passed.")
            else:
                try:
                    row, col = map(int, move_input.split())
                    if game.make_move(row, col):
                        print(f"You played at ({row}, {col})")
                    else:
                        print("Invalid move! Try again.")
                        continue
                except ValueError:
                    print("Invalid input! Use 'row col' format.")
                    continue
        
        else:  # AI player (white)
            print("AI is thinking...")
            ai_move = ai_player.get_move(game)
            
            if ai_move is None:
                game.pass_turn()
                print("AI passed.")
            else:
                row, col = ai_move
                if game.make_move(row, col):
                    print(f"AI played at ({row}, {col})")
                else:
                    # Fallback to random move
                    legal_moves = game.get_legal_moves()
                    if legal_moves:
                        row, col = random.choice(legal_moves)
                        game.make_move(row, col)
                        print(f"AI played at ({row}, {col}) [random fallback]")
                    else:
                        game.pass_turn()
                        print("AI passed.")
        
        print()
    
    print("Game over!")
    game.display_board()


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Play against dohungo AI")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return
    
    try:
        play_against_ai(model_path, device)
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 