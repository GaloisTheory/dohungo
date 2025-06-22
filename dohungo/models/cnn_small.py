"""Small CNN model for dohungo.

A compact convolutional neural network for predicting Go moves.
Target: ~450k parameters, 4 conv layers + 512-unit FC layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSmall(nn.Module):
    """Small CNN for Go move prediction.
    
    Architecture:
    - 4 convolutional layers with ReLU
    - Global average pooling or flatten
    - 512-unit fully connected layer
    - Output layer for 19x19 = 361 possible moves
    """
    
    def __init__(self, input_planes: int = 7, board_size: int = 19):
        """Initialize the CNN.
        
        Args:
            input_planes: Number of input feature planes.
            board_size: Size of the Go board (default 19).
        """
        super().__init__()
        self.input_planes = input_planes
        self.board_size = board_size
        self.output_size = board_size * board_size
        
        # Convolutional layers
        # Layer 1: input_planes -> 32 channels
        self.conv1 = nn.Conv2d(input_planes, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer 4: 128 -> 64 channels (bottleneck)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Calculate flattened size after convolutions
        # Since we're not using pooling, size remains 19x19
        self.flattened_size = 64 * board_size * board_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, self.output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_planes, board_size, board_size).
            
        Returns:
            Output tensor of shape (batch_size, board_size * board_size).
        """
        # Convolutional layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict_move(self, x: torch.Tensor) -> torch.Tensor:
        """Predict move probabilities using softmax.
        
        Args:
            x: Input tensor.
            
        Returns:
            Move probabilities of shape (batch_size, board_size * board_size).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return a summary string of the model architecture."""
        total_params = self.count_parameters()
        return (
            f"CNNSmall(\n"
            f"  Input: {self.input_planes} planes × {self.board_size}×{self.board_size}\n"
            f"  Conv layers: {self.input_planes}→32→64→128→64\n" 
            f"  FC layers: {self.flattened_size}→512→{self.output_size}\n"
            f"  Total parameters: {total_params:,}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the model
    model = CNNSmall(input_planes=7, board_size=19)
    print(model.summary())
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 7, 19, 19)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    probs = model.predict_move(x)
    print(f"Probabilities sum: {probs.sum(dim=1)}")  # Should be close to 1.0 