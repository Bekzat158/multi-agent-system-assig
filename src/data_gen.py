"""
Synthetic IIoT Dataset Generator
Generates multi-client non-IID time-series data for federated anomaly detection.
Each client simulates a different industrial edge node with distinct operating patterns.
"""

import numpy as np
import os


def set_seed(seed: int = 42):
    np.random.seed(seed)


def generate_client_data(
    client_id: int,
    n_samples: int = 5000,
    n_features: int = 6,
    anomaly_ratio: float = 0.05,
    seed: int = None,
) -> dict:
    """
    Generate synthetic IIoT time-series for one client.
    
    Non-IID: each client has different base frequency, amplitude, noise level.
    Anomalies: point anomalies (spikes), contextual (drift), collective (pattern shift).
    """
    rng = np.random.RandomState(seed if seed is not None else client_id * 100)

    # --- Client-specific parameters (creates non-IID distribution) ---
    base_freq = 0.01 + client_id * 0.008          # different periodicity
    amplitude = 1.0 + client_id * 0.4             # different signal strength
    noise_std = 0.05 + client_id * 0.03           # different noise level
    sensor_drift = rng.uniform(-0.2, 0.2, n_features)  # sensor bias per feature

    t = np.arange(n_samples)

    # --- Normal signal: sinusoidal + harmonics + noise ---
    X = np.zeros((n_samples, n_features))
    for f in range(n_features):
        phase = rng.uniform(0, 2 * np.pi)
        harmonic = rng.uniform(0.1, 0.4)
        X[:, f] = (
            amplitude * np.sin(2 * np.pi * base_freq * t + phase)
            + harmonic * amplitude * np.sin(4 * np.pi * base_freq * t + phase)
            + rng.normal(0, noise_std, n_samples)
            + sensor_drift[f]
        )

    # --- Inject anomalies ---
    labels = np.zeros(n_samples, dtype=int)
    n_anomalies = int(n_samples * anomaly_ratio)

    anomaly_types = ["spike", "drift", "pattern_shift"]
    anomaly_indices = []

    # 1. Point anomalies (spikes) — 50% of anomalies
    n_spikes = n_anomalies // 2
    spike_locs = rng.choice(np.arange(50, n_samples - 50), size=n_spikes, replace=False)
    for loc in spike_locs:
        spike_mag = rng.uniform(4.0, 8.0) * amplitude
        feat_idx = rng.randint(0, n_features)
        X[loc, feat_idx] += spike_mag * rng.choice([-1, 1])
        labels[loc] = 1
    anomaly_indices.extend(spike_locs.tolist())

    # 2. Drift anomalies — 30% of anomalies
    n_drift = n_anomalies * 3 // 10
    drift_starts = rng.choice(np.arange(100, n_samples - 100), size=max(1, n_drift // 20), replace=False)
    for ds in drift_starts:
        drift_len = rng.randint(15, 25)
        drift_len = min(drift_len, n_samples - ds)
        drift_mag = rng.uniform(2.5, 5.0) * amplitude
        for i in range(drift_len):
            X[ds + i] += drift_mag * (i / drift_len)
            labels[ds + i] = 1
        anomaly_indices.extend(list(range(ds, ds + drift_len)))

    # 3. Pattern shift — 20% of anomalies
    n_pattern = n_anomalies * 2 // 10
    pattern_starts = rng.choice(np.arange(200, n_samples - 200), size=max(1, n_pattern // 30), replace=False)
    for ps in pattern_starts:
        plen = rng.randint(20, 35)
        plen = min(plen, n_samples - ps)
        X[ps : ps + plen] *= rng.uniform(2.0, 3.5)
        labels[ps : ps + plen] = 1
        anomaly_indices.extend(list(range(ps, ps + plen)))

    return {
        "X": X,
        "labels": labels,
        "client_id": client_id,
        "n_features": n_features,
        "anomaly_ratio": labels.mean(),
        "params": {
            "base_freq": base_freq,
            "amplitude": amplitude,
            "noise_std": noise_std,
        },
    }


def create_train_test_split(data: dict, train_ratio: float = 0.7):
    """Split into train (mostly normal) and test (with anomalies) sets."""
    X, labels = data["X"], data["labels"]
    n = len(X)
    split = int(n * train_ratio)

    X_train = X[:split]
    y_train = labels[:split]
    X_test = X[split:]
    y_test = labels[split:]

    # For train: keep only normal samples (unsupervised AD scenario)
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    return {
        "X_train": X_train_normal,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_all": X_train,
        "y_train_all": y_train,
    }


def create_windows(X: np.ndarray, window_size: int = 30, stride: int = 1) -> np.ndarray:
    """Convert time-series to sliding windows."""
    n, f = X.shape
    windows = []
    for i in range(0, n - window_size + 1, stride):
        windows.append(X[i : i + window_size])
    return np.array(windows, dtype=np.float32)


def generate_all_clients(
    n_clients: int = 5,
    n_samples: int = 5000,
    n_features: int = 6,
    anomaly_ratio: float = 0.05,
    window_size: int = 30,
    save_dir: str = "data",
) -> list:
    """Generate dataset for all clients and save to disk."""
    os.makedirs(save_dir, exist_ok=True)
    set_seed(42)

    all_clients = []
    print(f"Generating synthetic IIoT dataset for {n_clients} clients...")

    for cid in range(n_clients):
        raw = generate_client_data(
            client_id=cid,
            n_samples=n_samples,
            n_features=n_features,
            anomaly_ratio=anomaly_ratio,
            seed=cid * 123 + 7,
        )
        split = create_train_test_split(raw, train_ratio=0.7)

        # Create windows
        X_train_w = create_windows(split["X_train"], window_size=window_size)
        X_test_w = create_windows(split["X_test"], window_size=window_size)
        y_test_w = split["y_test"][window_size - 1 :]  # label = last step of window

        client_data = {
            "client_id": cid,
            "X_train": X_train_w,
            "X_test": X_test_w,
            "y_test": y_test_w,
            "raw": raw,
            "params": raw["params"],
        }

        # Save
        np.save(os.path.join(save_dir, f"client_{cid}_train.npy"), X_train_w)
        np.save(os.path.join(save_dir, f"client_{cid}_test.npy"), X_test_w)
        np.save(os.path.join(save_dir, f"client_{cid}_labels.npy"), y_test_w)

        all_clients.append(client_data)
        print(
            f"  Client {cid}: train={X_train_w.shape}, test={X_test_w.shape}, "
            f"anomaly_ratio={raw['anomaly_ratio']:.3f}, freq={raw['params']['base_freq']:.4f}"
        )

    print(f"Dataset saved to {save_dir}/")
    return all_clients


if __name__ == "__main__":
    clients = generate_all_clients()
