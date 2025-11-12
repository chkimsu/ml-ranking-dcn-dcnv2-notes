"""
PyTorch version of the DCNv2 toy example.
Trains on data/sample_ranking_data.csv to demonstrate full-matrix cross layers.
"""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CrossLayerFull(nn.Module):
    """Full-rank cross layer identical to DCNv2 formulation."""

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x0, x):
        xw = x @ self.weight
        return x0 * (xw + self.bias) + x


class DCNv2(nn.Module):
    def __init__(self, input_dim, cross_layers=3, deep_units=(64, 32)):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayerFull(input_dim) for _ in range(cross_layers)])

        deep_layers = []
        in_dim = input_dim
        for units in deep_units:
            deep_layers.append(nn.Linear(in_dim, units))
            deep_layers.append(nn.ReLU())
            in_dim = units
        self.deep = nn.Sequential(*deep_layers)
        self.output = nn.Linear(input_dim + deep_units[-1], 1)

    def forward(self, x):
        x0 = x
        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        combined = torch.cat([cross, deep], dim=-1)
        return self.output(combined)


def load_dataset(csv_path, batch_size=4):
    df = pd.read_csv(csv_path)
    features = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    labels = torch.tensor(df[["label"]].values, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, loader, epochs=5, lr=5e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_ranking_data.csv"
    loader = load_dataset(data_path, batch_size=4)
    input_dim = next(iter(loader))[0].shape[-1]
    model = DCNv2(input_dim=input_dim, cross_layers=3, deep_units=(64, 32))
    train(model, loader, epochs=5)


if __name__ == "__main__":
    main()
