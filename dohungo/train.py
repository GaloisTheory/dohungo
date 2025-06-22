"""Training script for dohungo CNN model.

Usage:
    python -m dohungo.train --config configs/small.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from .data.dataloader import create_dataloader
from .data.encoders import SevenPlaneEncoder, SimpleEncoder
from .data.downloader import download_kgs_index
from .models.cnn_small import CNNSmall


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Neural network model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Calculate accuracy (top-1)
        _, predicted = torch.max(outputs.data, 1)
        _, target_indices = torch.max(targets.data, 1)
        correct_predictions += (predicted == target_indices).sum().item()
        total_predictions += targets.size(0)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return avg_loss, accuracy


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model.
    
    Args:
        model: Neural network model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, target_indices = torch.max(targets.data, 1)
            correct_predictions += (predicted == target_indices).sum().item()
            total_predictions += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return avg_loss, accuracy


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train dohungo CNN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")
    print(OmegaConf.to_yaml(config))
    
    # Set random seeds for reproducibility
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create directories
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Download data if needed
    if config.data.download:
        print("Downloading SGF files...")
        download_kgs_index(
            data_dir=data_dir,
            board_size=config.data.board_size,
            max_games=config.data.max_games,
        )
    
    # Create encoder
    encoder_name = config.model.encoder
    if encoder_name == "sevenplane":
        encoder = SevenPlaneEncoder()
    elif encoder_name == "simple":
        encoder = SimpleEncoder()
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")
    
    print(f"Using encoder: {encoder}")
    
    # Create model
    model = CNNSmall(
        input_planes=encoder.num_planes,
        board_size=config.data.board_size,
    )
    model = model.to(device)
    print(model.summary())
    
    # Create data loaders
    train_loader = create_dataloader(
        data_dir=data_dir,
        encoder=encoder,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        board_size=config.data.board_size,
    )
    
    # Create optimizer
    optimizer_name = config.training.optimizer.lower()
    if optimizer_name == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=config.training.learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config.training.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    best_model_path = checkpoint_dir / "best.pt"
    
    print(f"Starting training for {config.training.epochs} epochs...")
    
    for epoch in range(config.training.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate (using same loader for simplicity - in practice, use separate validation set)
        val_loss, val_acc = validate_model(model, train_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{config.training.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'config': config,
            }, best_model_path)
            print(f"  New best model saved! Accuracy: {best_accuracy:.4f}")
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main() 