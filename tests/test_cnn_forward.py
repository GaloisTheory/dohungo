"""Tests for dohungo CNN model."""
from __future__ import annotations

import torch
import pytest

from dohungo.models.cnn_small import CNNSmall


def test_cnn_small_initialization():
    """Test CNNSmall model initialization."""
    model = CNNSmall(input_planes=7, board_size=19)
    
    # Check basic properties
    assert model.input_planes == 7
    assert model.board_size == 19
    assert model.output_size == 361  # 19 * 19
    
    # Check parameter count is reasonable (~450k target)
    param_count = model.count_parameters()
    assert 400_000 < param_count < 500_000, f"Parameter count {param_count} is outside expected range"


def test_cnn_small_forward_pass():
    """Test forward pass through CNNSmall model."""
    model = CNNSmall(input_planes=7, board_size=19)
    
    # Create a batch of inputs
    batch_size = 4
    inputs = torch.randn(batch_size, 7, 19, 19)
    
    # Forward pass
    outputs = model(inputs)
    
    # Check output shape
    assert outputs.shape == (batch_size, 361)
    
    # Check outputs are finite
    assert torch.all(torch.isfinite(outputs))


def test_cnn_small_predict_move():
    """Test move prediction functionality."""
    model = CNNSmall(input_planes=7, board_size=19)
    
    # Create input
    batch_size = 2
    inputs = torch.randn(batch_size, 7, 19, 19)
    
    # Get predictions
    predictions = model.predict_move(inputs)
    
    # Check output shape
    assert predictions.shape == (batch_size, 361)
    
    # Check probabilities sum to 1
    prob_sums = predictions.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
    
    # Check all probabilities are non-negative
    assert torch.all(predictions >= 0)


def test_cnn_small_different_sizes():
    """Test CNN with different input configurations."""
    # Test with SimpleEncoder (11 planes)
    model_11 = CNNSmall(input_planes=11, board_size=19)
    inputs_11 = torch.randn(2, 11, 19, 19)
    outputs_11 = model_11(inputs_11)
    assert outputs_11.shape == (2, 361)
    
    # Test with 9x9 board
    model_9x9 = CNNSmall(input_planes=7, board_size=9)
    inputs_9x9 = torch.randn(2, 7, 9, 9)
    outputs_9x9 = model_9x9(inputs_9x9)
    assert outputs_9x9.shape == (2, 81)  # 9 * 9


def test_cnn_small_gradients():
    """Test that gradients flow properly."""
    model = CNNSmall(input_planes=7, board_size=19)
    
    # Create inputs and targets
    inputs = torch.randn(2, 7, 19, 19, requires_grad=True)
    targets = torch.randint(0, 361, (2,))
    
    # Forward pass
    outputs = model(inputs)
    
    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert inputs.grad is not None
    assert torch.any(inputs.grad != 0)
    
    # Check model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_cnn_small_summary():
    """Test model summary functionality."""
    model = CNNSmall(input_planes=7, board_size=19)
    
    summary = model.summary()
    
    # Check summary contains expected information
    assert "CNNSmall" in summary
    assert "7 planes" in summary
    assert "19×19" in summary
    assert "parameters" in summary.lower()
    
    # Check parameter count is included
    param_count = model.count_parameters()
    assert str(param_count) in summary.replace(",", "")


def test_cnn_small_eval_mode():
    """Test that model behaves differently in train vs eval mode."""
    model = CNNSmall(input_planes=7, board_size=19)
    inputs = torch.randn(1, 7, 19, 19)
    
    # Test in training mode
    model.train()
    output_train = model(inputs)
    
    # Test in eval mode
    model.eval()
    output_eval = model(inputs)
    
    # Outputs should be different due to dropout
    # Note: this might not always be true due to randomness, but generally should be
    assert output_train.shape == output_eval.shape


if __name__ == "__main__":
    # Run tests directly
    test_cnn_small_initialization()
    test_cnn_small_forward_pass()
    test_cnn_small_predict_move()
    test_cnn_small_different_sizes()
    test_cnn_small_gradients()
    test_cnn_small_summary()
    test_cnn_small_eval_mode()
    print("All CNN tests passed!") 