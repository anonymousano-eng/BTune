import torch
import torch.nn as nn
import csv
import numpy as np
from collections import deque
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List

class LSTM(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=64, window_size=30):
        super().__init__()
        self.window_size = window_size
        self.local_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.global_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.global_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 13)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 13)
        )

    def forward(self, x):
        if x.dim() != 3 or x.size(2) != 26:
            raise ValueError(f"Invalid input shape: {x.shape}")
        local_out, _ = self.local_lstm(x)
        global_out, (h_n, _) = self.global_lstm(local_out)
        global_state = torch.cat([h_n[0], h_n[1]], dim=-1)
        global_state = self.global_proj(global_state)
        attn_scores = torch.matmul(
            global_state.unsqueeze(1),
            local_out.transpose(1, 2)
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, local_out).squeeze(1)
        combined = torch.cat([global_state, context], dim=-1)
        projected = self.attn_layer(combined)
        return self.mean_head(projected), self.logvar_head(projected)

class DataProcessor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size + 5)
        self.global_scaler = StandardScaler()
        self.window_scaler = StandardScaler()
        self.global_fitted = False
        self.window_fitted = False
        self.model = LSTM(hidden_dim=64)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.train_step = 0
        self._init_logfile("system_log.csv")

    def _init_logfile(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",

                "pred_φcontext_switch", "pred_ζmemcont", "pred_σswap",
                "pred_δdisk_latency", "pred_ηnet_queue", "pred_ωiowait",
                "pred_ηdisk_queue", "pred_δnet", "pred_ρretrans",
                "pred_θcpu", "pred_τresponse_time", "pred_σnet_bandwidth",
                "pred_πthroughput",

                "actual_φcontext_switch", "actual_ζmemcont", "actual_σswap",
                "actual_δdisk_latency", "actual_ηnet_queue", "actual_ωiowait",
                "actual_ηdisk_queue", "actual_δnet", "actual_ρretrans",
                "actual_θcpu", "actual_τresponse_time", "actual_σnet_bandwidth",
                "actual_πthroughput",

                "confidence",
                "data_status", 
                "training"
            ])

    def _safe_normalize(self, data: np.ndarray) -> Optional[np.ndarray]:
        try:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.shape != (1, 13):
                raise ValueError(f"Invalid data shape: {data.shape}")
            if not self.global_fitted:
                self.global_scaler.partial_fit(data)
                self.global_fitted = True
            global_norm = self.global_scaler.transform(data)
            if len(self.data_buffer) >= 5:
                window_data = np.vstack(self.data_buffer)
                if not self.window_fitted:
                    self.window_scaler.partial_fit(window_data)
                    self.window_fitted = True
                normalized = self.window_scaler.transform(global_norm)
            else:
                normalized = global_norm
            return normalized.flatten()
        except Exception as e:
            print(f"Normalization failed: {str(e)}")
            return None

    def process_line(self, line: str) -> Optional[Tuple]:
        try:
            parts = line.strip().split()
            if len(parts) < 14:
                return None
            timestamp = parts[0]
            raw_data = np.array([float(x) for x in parts[1:14]])
            norm_data = self._safe_normalize(raw_data)
            if norm_data is None or norm_data.shape != (13,):
                print(f"Normalization result invalid: {norm_data}")
                return None
            self.data_buffer.append(norm_data.reshape(1, -1))
            if len(self.data_buffer) < self.window_size:
                return None
            window = np.vstack(list(self.data_buffer)[-self.window_size:])
            if window.ndim != 2 or window.shape[1] != 13:
                print(f"Invalid window shape: {window.shape}")
                return None
            if window.shape[0] < 3:
                return None
            valid_window = window[1:, :]
            diff_feat = np.diff(valid_window, axis=0)
            min_len = min(valid_window.shape[0] - 1, diff_feat.shape[0])
            aligned_window = valid_window[:min_len, :]
            aligned_diff = diff_feat[:min_len, :]
            final_feat = np.hstack([aligned_window, aligned_diff])
            tensor_data = torch.FloatTensor(final_feat).unsqueeze(0)
            self.train_step += 1
            return (timestamp, tensor_data, raw_data)
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            return None

    def update_model(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: List[float]):
        try:
            self.optimizer.zero_grad()
            target_tensor = torch.tensor([target], dtype=pred_mean.dtype, device=pred_mean.device)
            var = torch.exp(pred_logvar)
            loss = 0.5 * torch.sum(pred_logvar + (target_tensor - pred_mean)**2 / var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.train_step % 100 == 0:
                torch.save(self.model.state_dict(), f"model_{self.train_step}.pt")
        except Exception as e:
            print(f"Update failed: {str(e)}")

    def log_data(self, ts: str, pred_mean: List[float], actual: List[float], pred_logvar: List[float], status: str):
        try:
            confidence_values = [1 / (1 + np.exp(-lv)) for lv in pred_logvar]
            avg_confidence = np.mean(confidence_values)
            with open("system_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                row = [ts]
                row.extend([f"{p:.4f}" for p in pred_mean])
                row.extend([f"{a:.2f}" for a in actual])
                row.append(f"{avg_confidence:.4f}")
                row.extend([status, "Y" if self.model.training else "N"])
                writer.writerow(row)
        except Exception as e:
            print(f"Logging failed: {str(e)}")


if __name__ == "__main__":
    processor = DataProcessor()
    sample_input = "2023-10-01T12:00:00 " + " ".join(["0.1"]*13)
    processed = processor.process_line(sample_input)
    if processed:
        timestamp, tensor_data, actual = processed
        mean, logvar = processor.model(tensor_data)
        processor.update_model(mean, logvar, actual)
        processor.log_data(timestamp, 
                          mean.squeeze().tolist(), 
                          actual.tolist(), 
                          logvar.squeeze().tolist(), 
                          "valid")