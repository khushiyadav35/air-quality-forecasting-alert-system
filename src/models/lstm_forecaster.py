"""
PyTorch Deep LSTM Recurrent Neural Network for Air Quality Forecasting
Implements sequence-to-sequence multi-horizon (24h to 72h) forecasting
Author: Khushi Yadav (@khushiyadav35)
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class AQILSTMNet(nn.Module):
    """
    Stacked Deep LSTM Recurrent Architecture for Atmospheric Sequence Modeling.
    Features:
    - Multi-layer LSTM recurrent encoder
    - Dropout regularization to mitigate overfitting on seasonal artifacts
    - Dense projection head for multi-step ahead continuous prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        dropout: float = 0.2,
    ):
        super(AQILSTMNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use final recurrent hidden state representation
        last_hidden = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        out = self.fc_head(last_hidden)   # shape: (batch_size, forecast_horizon)
        return out


class LSTMTrainer:
    """Handles PyTorch training loops, loss functions, and inference."""

    def __init__(
        self,
        input_dim: int,
        forecast_horizon: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.forecast_horizon = forecast_horizon
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AQILSTMNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            dropout=0.2,
        ).to(self.device)

        # Smooth L1 / Huber Loss is robust to extreme winter smog outliers
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.residual_std: float = 12.0  # Empirical residual variance for confidence bounds

    def train_epochs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 64,
    ) -> Dict[str, list]:
        """Runs iterative gradient descent with validation loss tracking."""
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_train_loss += loss.item() * len(batch_x)

            epoch_train_loss = total_train_loss / len(train_dataset)

            # Validation step
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for v_x, v_y in val_loader:
                    v_x, v_y = v_x.to(self.device), v_y.to(self.device)
                    v_pred = self.model(v_x)
                    v_loss = self.criterion(v_pred, v_y)
                    total_val_loss += v_loss.item() * len(v_x)

            epoch_val_loss = total_val_loss / len(val_dataset)
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = self.model.state_dict().copy()

            self.model.train()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Estimate residual uncertainty on validation set
        self.model.eval()
        with torch.no_grad():
            val_preds = self.predict(X_val)
            residuals = y_val - val_preds
            self.residual_std = float(np.std(residuals))

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference for scaled inputs."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).float().to(self.device)
            preds = self.model(x_tensor)
            return preds.cpu().numpy()

    def predict_with_intervals(
        self,
        X_seq: np.ndarray,
        feature_engineer,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates forecast trajectory with 95% confidence intervals on actual NAQI scale:
        returns: (y_hat, y_lower, y_upper)
        """
        scaled_preds = self.predict(X_seq)
        # Invert scaling
        y_hat = feature_engineer.inverse_transform_target(scaled_preds)

        # Estimate unscaled standard error
        unscaled_sigma = self.residual_std * (
            feature_engineer.target_scaler.data_range_[0]
            if hasattr(feature_engineer.target_scaler, "data_range_")
            else 400.0
        )
        # Horizon growth factor: uncertainty expands forward in time
        growth = np.linspace(1.0, 1.6, self.forecast_horizon)
        y_lower = np.clip(y_hat - 1.96 * unscaled_sigma * growth, a_min=10.0, a_max=None)
        y_upper = y_hat + 1.96 * unscaled_sigma * growth

        return y_hat, y_lower, y_upper

    def save_checkpoint(self, path: Path) -> None:
        """Serializes weights and architecture metadata."""
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "forecast_horizon": self.model.forecast_horizon,
                "residual_std": self.residual_std,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Path, device: Optional[str] = None) -> "LSTMTrainer":
        """Loads serialized checkpoint."""
        ckpt = torch.load(path, map_location=torch.device(device if device else "cpu"))
        trainer = cls(
            input_dim=ckpt["input_dim"],
            forecast_horizon=ckpt["forecast_horizon"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            device=device,
        )
        trainer.model.load_state_dict(ckpt["state_dict"])
        trainer.residual_std = ckpt.get("residual_std", 12.0)
        trainer.model.eval()
        return trainer
