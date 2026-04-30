"""
LSTM / GRU Time-Series Forecasting — Pure NumPy implementation
No TensorFlow/PyTorch required. Works on any Python version.

Architecture: single-layer LSTM cell + dense output
Training: mini-batch gradient descent with Adam optimizer, early stopping
"""

import os, sys, io, pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluation import regression_metrics, plot_predicted_vs_actual

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "chennai_groundwater.csv")

SEQUENCE_LEN   = 12
FORECAST_STEPS = 3
EPOCHS         = 80
BATCH_SIZE     = 16
HIDDEN_SIZE    = 32
LEARNING_RATE  = 0.001
PATIENCE       = 10


# ── Activation helpers ────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def tanh(x):
    return np.tanh(np.clip(x, -30, 30))

def sigmoid_grad(s):   return s * (1 - s)
def tanh_grad(t):      return 1 - t ** 2


# ── Adam optimiser state ──────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps
        self.m = {}; self.v = {}; self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


# ── Pure NumPy LSTM ───────────────────────────────────────────────────────────

class NumpyLSTM:
    """
    Single-layer LSTM followed by a linear dense layer.
    Input shape : (batch, seq_len, n_features)
    Output shape: (batch, n_steps)
    """

    def __init__(self, n_features: int, hidden: int, n_steps: int,
                 lr: float = LEARNING_RATE):
        H, F, S = hidden, n_features, n_steps
        scale = 0.05

        # LSTM gate weights  [input_gate, forget_gate, cell_gate, output_gate]
        self.Wx = np.random.randn(4 * H, F) * scale   # input → gates
        self.Wh = np.random.randn(4 * H, H) * scale   # hidden → gates
        self.b  = np.zeros(4 * H)

        # Dense output layer
        self.Wy = np.random.randn(S, H) * scale
        self.by = np.zeros(S)

        self.H = H
        self.optimizer = Adam(lr=lr)

    @property
    def params(self):
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b,
                "Wy": self.Wy, "by": self.by}

    def _set_params(self, p):
        self.Wx = p["Wx"]; self.Wh = p["Wh"]; self.b = p["b"]
        self.Wy = p["Wy"]; self.by = p["by"]

    def _lstm_forward_single(self, x_seq):
        """
        x_seq: (seq_len, n_features)  — one sample
        Returns: last hidden state h  (hidden,)
                 cache for backprop
        """
        H   = self.H
        T   = x_seq.shape[0]
        h   = np.zeros(H)
        c   = np.zeros(H)
        cache = []

        for t in range(T):
            gates_raw = self.Wx @ x_seq[t] + self.Wh @ h + self.b
            i_raw, f_raw, g_raw, o_raw = np.split(gates_raw, 4)

            i = sigmoid(i_raw)
            f = sigmoid(f_raw)
            g = tanh(g_raw)
            o = sigmoid(o_raw)

            c_new = f * c + i * g
            h_new = o * tanh(c_new)

            cache.append((x_seq[t], h, c, i, f, g, o, c_new, h_new))
            h, c = h_new, c_new

        return h, cache

    def forward(self, X_batch):
        """X_batch: (batch, seq, features) → y_hat: (batch, n_steps)"""
        batch_h = []
        caches  = []
        for b in range(X_batch.shape[0]):
            h, cache = self._lstm_forward_single(X_batch[b])
            batch_h.append(h)
            caches.append(cache)
        H_mat   = np.stack(batch_h)          # (batch, H)
        y_hat   = H_mat @ self.Wy.T + self.by  # (batch, n_steps)
        return y_hat, H_mat, caches

    def backward(self, X_batch, y_true, y_hat, H_mat, caches):
        B, S_out = y_hat.shape
        H = self.H

        # Dense layer gradients
        dL_dy   = 2 * (y_hat - y_true) / B         # (B, S_out)
        dWy     = dL_dy.T @ H_mat                   # (S_out, H)
        dby     = dL_dy.sum(axis=0)
        dH_mat  = dL_dy @ self.Wy                   # (B, H)

        # Accumulate LSTM weight grads across batch
        dWx_acc = np.zeros_like(self.Wx)
        dWh_acc = np.zeros_like(self.Wh)
        db_acc  = np.zeros_like(self.b)

        for b in range(B):
            dh_next = dH_mat[b]
            dc_next = np.zeros(H)
            cache   = caches[b]

            for t in reversed(range(len(cache))):
                x_t, h_prev, c_prev, i, f, g, o, c_new, h_new = cache[t]

                tanh_c = tanh(c_new)
                do = dh_next * tanh_c
                dc = dh_next * o * tanh_grad(tanh_c) + dc_next
                di = dc * g
                df = dc * c_prev
                dg = dc * i
                dc_next = dc * f

                di_raw = di * sigmoid_grad(i)
                df_raw = df * sigmoid_grad(f)
                dg_raw = dg * tanh_grad(g)
                do_raw = do * sigmoid_grad(o)

                dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw])
                dWx_acc += np.outer(dgates, x_t)
                dWh_acc += np.outer(dgates, h_prev)
                db_acc  += dgates
                dh_next  = self.Wh.T @ dgates

        # Clip gradients
        for g in [dWx_acc, dWh_acc, db_acc, dWy, dby]:
            np.clip(g, -5, 5, out=g)

        grads = {"Wx": dWx_acc, "Wh": dWh_acc, "b": db_acc,
                 "Wy": dWy,     "by": dby}
        return grads

    def train_epoch(self, X, y, batch_size):
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        total_loss = 0.0
        for start in range(0, len(X), batch_size):
            Xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            y_hat, H_mat, caches = self.forward(Xb)
            loss = np.mean((y_hat - yb) ** 2)
            total_loss += loss * len(Xb)
            grads = self.backward(Xb, yb, y_hat, H_mat, caches)
            p = self.params
            p = self.optimizer.step(p, grads)
            self._set_params(p)
        return total_loss / len(X)

    def predict(self, X):
        y_hat, _, _ = self.forward(X)
        return y_hat

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


# ── Sequence creation ─────────────────────────────────────────────────────────

def make_sequences(data: np.ndarray, seq_len: int, n_steps: int):
    X, y = [], []
    for i in range(len(data) - seq_len - n_steps + 1):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + n_steps, 0])  # target col 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Zone training ─────────────────────────────────────────────────────────────

def train_zone_model(zone_df: pd.DataFrame, zone_name: str,
                     model_type: str = "LSTM",
                     feature_cols: list = None):
    zone_df = zone_df.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = ["groundwater_level", "rainfall_mm", "temperature_c"]

    # groundwater_level must be first (target col 0)
    cols = ["groundwater_level"] + [c for c in feature_cols if c != "groundwater_level"]
    data = zone_df[cols].values.astype(np.float32)

    scaler = MinMaxScaler()
    data_sc = scaler.fit_transform(data)

    split = int(len(data_sc) * 0.8)
    train_data = data_sc[:split]
    test_data  = data_sc[max(0, split - SEQUENCE_LEN):]

    X_train, y_train = make_sequences(train_data, SEQUENCE_LEN, FORECAST_STEPS)
    X_test,  y_test  = make_sequences(test_data,  SEQUENCE_LEN, FORECAST_STEPS)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"    Skipping {zone_name} — not enough sequences")
        return None, scaler, {"model": f"{model_type}_{zone_name}"}, {}

    n_features = X_train.shape[2]
    model = NumpyLSTM(n_features=n_features, hidden=HIDDEN_SIZE,
                      n_steps=FORECAST_STEPS, lr=LEARNING_RATE)

    # Validation split from train (last 15%)
    val_cut = int(len(X_train) * 0.85)
    X_val, y_val = X_train[val_cut:], y_train[val_cut:]
    X_tr,  y_tr  = X_train[:val_cut], y_train[:val_cut]

    best_val, best_weights, patience_cnt = np.inf, None, 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        tr_loss  = model.train_epoch(X_tr, y_tr, BATCH_SIZE)
        val_pred = model.predict(X_val)
        val_loss = np.mean((val_pred - y_val) ** 2)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_weights = pickle.dumps(model)   # in-memory snapshot
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}  val_loss={best_val:.6f}")
                break

    # Restore best weights
    model = pickle.loads(best_weights)

    # Evaluate
    y_pred_sc = model.predict(X_test)[:, 0]   # first forecast step

    def inv(arr):
        dummy = np.zeros((len(arr), data.shape[1]))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_inv = inv(y_pred_sc)
    y_test_inv = inv(y_test[:, 0])

    metrics = regression_metrics(y_test_inv, y_pred_inv,
                                 label=f"{model_type}_{zone_name}")
    plot_predicted_vs_actual(y_test_inv, y_pred_inv,
                             f"{model_type}_{zone_name.replace(' ', '_')}", OUTPUT_DIR)
    _plot_training_history(train_losses, val_losses, zone_name, model_type)

    save_path = os.path.join(MODEL_DIR,
                             f"{model_type}_{zone_name.replace(' ', '_')}.pkl")
    model.save(save_path)
    joblib.dump(scaler, os.path.join(MODEL_DIR,
                f"{model_type}_{zone_name.replace(' ', '_')}_scaler.pkl"))

    history = {"loss": train_losses, "val_loss": val_losses}
    return model, scaler, metrics, history


def _plot_training_history(train_losses, val_losses, zone_name, model_type):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_title(f"{model_type} Training — {zone_name}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f"{model_type}_{zone_name.replace(' ', '_')}_training.png")
    plt.savefig(path, dpi=120)
    plt.close()


def forecast_future(model, last_sequence: np.ndarray, scaler,
                    feature_cols: list, n_months: int = 12) -> np.ndarray:
    """Autoregressively forecast n_months ahead."""
    preds = []
    seq   = last_sequence.copy()

    steps = (n_months + FORECAST_STEPS - 1) // FORECAST_STEPS
    for _ in range(steps):
        x_in  = seq[-SEQUENCE_LEN:].reshape(1, SEQUENCE_LEN, seq.shape[1])
        y_out = model.predict(x_in)[0]
        for step_val in y_out:
            new_row    = seq[-1].copy()
            new_row[0] = step_val
            seq        = np.vstack([seq, new_row])
            preds.append(step_val)

    dummy = np.zeros((len(preds), seq.shape[1]))
    dummy[:, 0] = preds
    return scaler.inverse_transform(dummy)[:n_months, 0]


def run_pipeline(data_path: str = DATA_PATH, model_type: str = "LSTM",
                 zones: list = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"])
    if zones is None:
        zones = df["zone"].unique().tolist()

    feature_cols = ["groundwater_level", "rainfall_mm", "temperature_c",
                    "evapotranspiration", "ndvi", "urban_fraction"]

    all_metrics = []
    for zone in zones:
        print(f"\n  Training {model_type} for zone: {zone}")
        zdf = df[df["zone"] == zone].copy()
        if len(zdf) < SEQUENCE_LEN * 3:
            print(f"    Skipping {zone} (insufficient data)")
            continue
        _, _, metrics, _ = train_zone_model(zdf, zone, model_type, feature_cols)
        if metrics:
            all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    print(f"\n=== {model_type} Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, f"{model_type}_metrics.csv"), index=False)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["LSTM", "GRU"], default="LSTM")
    parser.add_argument("--zones", nargs="+", default=None)
    args = parser.parse_args()
    run_pipeline(model_type=args.model, zones=args.zones)
