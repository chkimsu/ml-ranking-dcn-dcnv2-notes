"""
PyTorch implementation of the ESMM architecture trained on sample_esmm_data.csv.
Shows shared-bottom CTR/CVR multi-task learning with tiny synthetic data.
"""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_mlp(input_dim, units):
    layers = []
    in_dim = input_dim
    for u in units:
        layers.append(nn.Linear(in_dim, u))
        layers.append(nn.ReLU())
        in_dim = u
    return nn.Sequential(*layers), in_dim


class ESMM(nn.Module):
    def __init__(self, input_dim, shared_units=(64,), tower_units=(32, 16)):
        super().__init__()
        self.shared_bottom, shared_dim = make_mlp(input_dim, shared_units)
        self.ctr_tower, ctr_dim = make_mlp(shared_dim, tower_units)
        self.cvr_tower, cvr_dim = make_mlp(shared_dim, tower_units)
        self.ctr_output = nn.Linear(ctr_dim, 1)
        self.cvr_output = nn.Linear(cvr_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shared = self.shared_bottom(x)
        ctr = self.ctr_tower(shared)
        cvr = self.cvr_tower(shared)
        ctr_prob = self.sigmoid(self.ctr_output(ctr))
        cvr_prob = self.sigmoid(self.cvr_output(cvr))
        ctcvr_prob = ctr_prob * cvr_prob
        return ctr_prob, cvr_prob, ctcvr_prob


def load_dataset(csv_path, batch_size=4):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("click", "conversion")]
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    click = torch.tensor(df[["click"]].values, dtype=torch.float32)
    conversion = torch.tensor(df[["conversion"]].values, dtype=torch.float32)
    dataset = TensorDataset(features, click, conversion)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, len(feature_cols)


def train(model, loader, epochs=8, lr=5e-3):
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        ctr_loss_total = 0.0
        ctcvr_loss_total = 0.0
        for features, click, conversion in loader:
            ctr_prob, _, ctcvr_prob = model(features)
            ctr_loss = bce(ctr_prob, click)
            ctcvr_loss = bce(ctcvr_prob, conversion)
            loss = ctr_loss + ctcvr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ctr_loss_total += ctr_loss.item()
            ctcvr_loss_total += ctcvr_loss.item()
        size = len(loader)
        print(
            f"Epoch {epoch+1}: CTR loss={ctr_loss_total / size:.4f}, "
            f"CTCVR loss={ctcvr_loss_total / size:.4f}"
        )


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_esmm_data.csv"
    loader, input_dim = load_dataset(data_path, batch_size=4)
    model = ESMM(input_dim=input_dim, shared_units=(64,), tower_units=(32, 16))
    train(model, loader, epochs=8)


if __name__ == "__main__":
    main()
