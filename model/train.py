"""
Training script for Multimodal Flood U-Net.
Trains on Kuro Siwo dataset with CrossEntropy + Dice Loss.
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.unet import FloodUNet
from model.dataset import FloodDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model.train")

def train_model(epochs: int = 5, batch_size: int = 8, lr: float = 1e-3, save_path: str = "model/weights/kuro_siwo_unet.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    train_dataset = FloodDataset(split="train")
    val_dataset = FloodDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FloodUNet(in_channels=7, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 5.0]).to(device)) # Weight flood class higher
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = (correct / total) * 100
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--weights", type=str, default="model/weights/kuro_siwo_unet.pt")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, save_path=args.weights)
