"""
Anomaly Detection Models for IIoT Federated Learning
- LSTMAutoencoder: reconstruction-based anomaly detection
- DeepSVDD: one-class classification (lightweight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────── LSTM Autoencoder ────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, latent_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        # x: (batch, window_size, n_features)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # last layer hidden state
        z = self.fc(h)
        return z


class LSTMDecoder(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, n_features: int, window_size: int, num_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, z):
        # z: (batch, latent_size)
        h = self.fc(z)  # (batch, hidden_size)
        h = h.unsqueeze(1).repeat(1, self.window_size, 1)  # (batch, window, hidden)
        out, _ = self.lstm(h)
        recon = self.out(out)  # (batch, window, n_features)
        return recon


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection.
    Anomaly score = mean squared reconstruction error per sample.
    """

    def __init__(
        self,
        n_features: int = 6,
        window_size: int = 30,
        hidden_size: int = 64,
        latent_size: int = 16,
        num_layers: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = LSTMEncoder(n_features, hidden_size, latent_size, num_layers)
        self.decoder = LSTMDecoder(latent_size, hidden_size, n_features, window_size, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def reconstruction_error(self, x):
        """Per-sample mean squared reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            err = F.mse_loss(recon, x, reduction="none")
            # mean over window and features
            score = err.mean(dim=[1, 2])
        return score.cpu().numpy()


# ─────────────────────────── Deep SVDD (lightweight) ─────────────────────────

class DeepSVDD(nn.Module):
    """
    Deep SVDD for one-class anomaly detection.
    Maps inputs to a hypersphere; anomaly score = distance from center.
    Uses 1D-CNN + FC encoder (lighter than LSTM for resource-constrained devices).
    """

    def __init__(self, n_features: int = 6, window_size: int = 30, latent_size: int = 16):
        super().__init__()
        self.latent_size = latent_size
        self.center = None  # set after warm-up training

        # 1D-CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64 * 4, latent_size),
        )

    def forward(self, x):
        # x: (batch, window, features) → transpose to (batch, features, window)
        x_t = x.permute(0, 2, 1)
        z = self.encoder(x_t)
        return z

    def anomaly_score(self, x):
        """Distance from center = anomaly score."""
        with torch.no_grad():
            z = self.forward(x)
            if self.center is None:
                raise ValueError("Center not initialized. Run init_center first.")
            dist = torch.sum((z - self.center) ** 2, dim=1)
        return dist.cpu().numpy()

    def init_center(self, dataloader, device, eps=0.1):
        """Initialize hypersphere center as mean of embeddings."""
        self.eval()
        z_all = []
        with torch.no_grad():
            for (xb,) in dataloader:
                xb = xb.to(device)
                z = self.forward(xb)
                z_all.append(z)
        center = torch.cat(z_all, dim=0).mean(dim=0)
        # Avoid center collapse
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center >= 0)] = eps
        self.center = center


# ─────────────────────────── Training utilities ───────────────────────────────

def train_autoencoder(
    model: LSTMAutoencoder,
    X_train: np.ndarray,
    n_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> list:
    """Train LSTM autoencoder. Returns list of epoch losses."""
    model = model.to(device)
    model.train()

    tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg_loss = epoch_loss / len(tensor)
        losses.append(avg_loss)
        if verbose:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}")

    return losses


def train_svdd(
    model: DeepSVDD,
    X_train: np.ndarray,
    n_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> list:
    """Train Deep SVDD. Returns list of epoch losses."""
    model = model.to(device)

    tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    # Initialize center
    model.eval()
    model.init_center(loader, device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            z = model(xb)
            dist = torch.sum((z - model.center) ** 2, dim=1)
            loss = dist.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg_loss = epoch_loss / len(tensor)
        losses.append(avg_loss)
        if verbose:
            print(f"    Epoch {epoch+1}/{n_epochs}: svdd_loss={avg_loss:.6f}")

    return losses


def compute_anomaly_scores(model, X_test: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Compute anomaly scores for test set."""
    model.eval()
    model = model.to(device)
    tensor = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    scores = []
    if isinstance(model, LSTMAutoencoder):
        for (xb,) in loader:
            xb = xb.to(device)
            s = model.reconstruction_error(xb)
            scores.append(s)
    elif isinstance(model, DeepSVDD):
        for (xb,) in loader:
            xb = xb.to(device)
            s = model.anomaly_score(xb)
            scores.append(s)

    return np.concatenate(scores)


def get_threshold(scores_train: np.ndarray, percentile: float = 95.0) -> float:
    """Compute anomaly threshold from training reconstruction errors."""
    return float(np.percentile(scores_train, percentile))
