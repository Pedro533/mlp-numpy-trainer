# MLP from scratch (NumPy) with Adam, Early Stopping, LR decay, L2 and Dropout
# Demo on make_moons dataset. Saves weights and scaler; includes helpers for loading/inference.
# Requirements: numpy, scikit-learn, joblib, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------- Utilities ----------
def one_hot(y, num_classes):
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1
    return Y

def relu(z):  return np.maximum(0, z)
def drelu(a): return (a > 0).astype(a.dtype)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(probs, Y_true_onehot, eps=1e-12):
    return -np.mean(np.sum(Y_true_onehot * np.log(probs + eps), axis=1))

# ---------- Visualization ----------
def plot_data_moons(X, y, title="Moons dataset", fname=None):
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=160)
    plt.show()

def plot_decision_boundary(mlp, scaler, X_raw, y, title="Decision boundary", fname=None, res=300):
    # Meshgrid in original coordinates (before scaling)
    x_min, x_max = X_raw[:,0].min()-0.5, X_raw[:,0].max()+0.5
    y_min, y_max = X_raw[:,1].min()-0.5, X_raw[:,1].max()+0.5
    n_side = int(np.sqrt(res**2))
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_side),
                         np.linspace(y_min, y_max, n_side))
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    # Normalize and pass through MLP
    grid = scaler.transform(grid_raw)
    _, a_s, _ = mlp.forward(grid, training=False)
    probs = a_s[-1][:,1].reshape(xx.shape)  # probability of class 1

    plt.figure(figsize=(5,5))
    cs = plt.contourf(xx, yy, probs, levels=30, alpha=0.8)
    plt.colorbar(cs)
    plt.scatter(X_raw[:,0], X_raw[:,1], c=y, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout()
    if fname: plt.savefig(fname, dpi=160)
    plt.show()

def plot_loss_curves(train_hist, val_hist=None, title="Loss curves"):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    if val_hist is not None: plt.plot(val_hist, label="val")
    plt.title(title); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- MLP ----------
class MLP:
    def __init__(self, layer_sizes, l2=0.0, dropout_p=0.0, seed=0):
        """
        layer_sizes: [n_in, h1, ..., n_out]
        l2: L2 weight decay
        dropout_p: dropout probability in hidden layers (0 = no dropout)
        """
        self.rng = np.random.default_rng(seed)
        self.L = len(layer_sizes) - 1
        self.l2 = l2
        self.dropout_p = dropout_p
        self.params = []
        for i in range(self.L):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            # He init for hidden layers, Xavier simple for output
            scale = np.sqrt(2.0 / fan_in) if i < self.L - 1 else np.sqrt(1.0 / fan_in)
            W = self.rng.normal(0.0, scale, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.params.append({"W": W, "b": b})
        # Adam states
        self.opt = [{"mW": np.zeros_like(p["W"]), "vW": np.zeros_like(p["W"]),
                     "mb": np.zeros_like(p["b"]), "vb": np.zeros_like(p["b"])} for p in self.params]
        # Loss history (for plotting)
        self.train_loss_hist = []
        self.val_loss_hist = []

    def _apply_dropout(self, a):
        """Inverted dropout (scaled by 1/(1-p) during training)."""
        if self.dropout_p <= 0:
            return a, None
        mask = (self.rng.random(a.shape) > self.dropout_p).astype(a.dtype) / (1.0 - self.dropout_p)
        return a * mask, mask

    def forward(self, X, training=False):
        """Returns (z_s, a_s, masks). a_s[0] = X"""
        z_s, a_s, masks = [], [X], []
        for i in range(self.L):
            W, b = self.params[i]["W"], self.params[i]["b"]
            z = a_s[-1] @ W + b
            z_s.append(z)
            if i < self.L - 1:
                a = relu(z)
                if training:
                    a, mask = self._apply_dropout(a)
                else:
                    mask = None
                masks.append(mask)
            else:
                a = softmax(z)
                masks.append(None)
            a_s.append(a)
        return z_s, a_s, masks

    def compute_loss(self, X, y):
        _, a_s, _ = self.forward(X, training=False)
        probs = a_s[-1]
        Y = one_hot(y, probs.shape[1])
        data_loss = cross_entropy(probs, Y)
        l2_loss = 0.0
        if self.l2 > 0:
            for p in self.params:
                l2_loss += 0.5 * self.l2 * np.sum(p["W"]**2)
            l2_loss /= X.shape[0]
        return data_loss + l2_loss

    def backward(self, z_s, a_s, masks, y):
        grads = []
        Y = one_hot(y, a_s[-1].shape[1])
        delta = (a_s[-1] - Y)                      # (N, C)
        N = y.shape[0]
        for i in reversed(range(self.L)):
            a_prev = a_s[i]
            dW = (a_prev.T @ delta) / N
            db = np.mean(delta, axis=0, keepdims=True)
            if self.l2 > 0:
                dW += (self.l2 / N) * self.params[i]["W"]
            grads.insert(0, {"dW": dW, "db": db})
            if i > 0:
                W = self.params[i]["W"]
                delta = (delta @ W.T) * drelu(a_s[i])
                if masks[i-1] is not None:
                    delta = delta * masks[i-1]
        return grads

    def _adam_step(self, grads, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for i in range(self.L):
            gW, gb = grads[i]["dW"], grads[i]["db"]
            o = self.opt[i]
            o["mW"] = beta1 * o["mW"] + (1 - beta1) * gW
            o["vW"] = beta2 * o["vW"] + (1 - beta2) * (gW * gW)
            o["mb"] = beta1 * o["mb"] + (1 - beta1) * gb
            o["vb"] = beta2 * o["vb"] + (1 - beta2) * (gb * gb)
            mW_hat = o["mW"] / (1 - beta1**t); vW_hat = o["vW"] / (1 - beta2**t)
            mb_hat = o["mb"] / (1 - beta1**t); vb_hat = o["vb"] / (1 - beta2**t)
            self.params[i]["W"] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.params[i]["b"] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def fit(self, X, y, X_val=None, y_val=None,
            lr=0.01, epochs=400, batch_size=128, print_every=10,
            patience=25, lr_decay=0.005):
        """
        Training with Adam + early stopping.
        lr_decay: k in schedule lr_t = lr_0 / (1 + k*epoch)  (can be left 0 with Adam)
        """
        N = X.shape[0]
        idx = np.arange(N)

        best_val = np.inf
        best_params = None
        waits = 0
        lr0 = lr
        t = 0  # Adam step counter

        for ep in range(1, epochs + 1):
            # LR decay per epoch (optional with Adam)
            lr = lr0 / (1.0 + lr_decay * ep) if lr_decay > 0 else lr0

            self.rng.shuffle(idx)
            for k in range(0, N, batch_size):
                batch = idx[k:k+batch_size]
                Xb, yb = X[batch], y[batch]
                z_s, a_s, masks = self.forward(Xb, training=True)
                grads = self.backward(z_s, a_s, masks, yb)
                t += 1
                self._adam_step(grads, lr, t)

            # logging + history
            tr = self.compute_loss(X, y)
            va = self.compute_loss(X_val, y_val) if X_val is not None else tr
            self.train_loss_hist.append(tr)
            self.val_loss_hist.append(va if X_val is not None else tr)
            if ep % print_every == 0 or ep == 1:
                print(f"Epoch {ep:04d} | lr={lr:.4f} | train_loss={tr:.4f} | val_loss={va:.4f}")

            # early stopping
            if X_val is not None:
                val_loss = va
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    waits = 0
                    best_params = [{"W": p["W"].copy(), "b": p["b"].copy()} for p in self.params]
                else:
                    waits += 1
                    if waits >= patience:
                        if best_params is not None:
                            self.params = best_params
                        print(f"Early stopping at epoch {ep} | best val_loss={best_val:.4f}")
                        break

    def predict(self, X):
        _, a_s, _ = self.forward(X, training=False)
        return np.argmax(a_s[-1], axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ---------- Production helpers ----------
def load_model(weights_path="mlp_weights.npz", scaler_path="scaler.joblib",
               arch=(2, 128, 64, 2), l2=0.0, dropout_p=0.0, seed=42):
    mlp = MLP(list(arch), l2=l2, dropout_p=dropout_p, seed=seed)
    data = np.load(weights_path)
    for i, p in enumerate(mlp.params):
        p["W"] = data[f"W{i}"]
        p["b"] = data[f"b{i}"]
    scaler = joblib.load(scaler_path)
    return mlp, scaler

def predict_proba(mlp, scaler, X_raw):
    Xn = scaler.transform(np.asarray(X_raw))
    _, a_s, _ = mlp.forward(Xn, training=False)
    return a_s[-1]  # softmax probabilities

# ---------- Demo ----------
def run_demo():
    print("== Training start ==")
    X_raw, y = make_moons(n_samples=2000, noise=0.25, random_state=0)

    # 1) Visualize raw data (two moons)
    plot_data_moons(X_raw, y, title="Moons dataset (raw)")

    # Split raw (before scaling) for consistent plots
    X_train_raw, X_tmp_raw, y_train, y_tmp = train_test_split(
        X_raw, y, test_size=0.4, random_state=0, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_tmp_raw, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp
    )

    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    # Architecture: 2 -> 128 -> 64 -> 2
    mlp = MLP(layer_sizes=[2, 128, 64, 2], l2=1e-3, dropout_p=0.05, seed=42)

    # 2) Decision boundary before training
    plot_decision_boundary(mlp, scaler, X_raw, y, title="Decision boundary (before training)")

    # 3) Train
    mlp.fit(
        X_train, y_train, X_val, y_val,
        lr=0.01, epochs=400, batch_size=128,
        print_every=10, patience=25, lr_decay=0.005
    )

    # 4) Loss curves
    plot_loss_curves(mlp.train_loss_hist, mlp.val_loss_hist, title="Loss (train/val)")

    # 5) Decision boundary after training
    plot_decision_boundary(mlp, scaler, X_raw, y, title="Decision boundary (after training)")

    # Metrics
    print("Train acc:", round(mlp.accuracy(X_train, y_train), 4))
    print("Val   acc:", round(mlp.accuracy(X_val, y_val), 4))
    print("Test  acc:", round(mlp.accuracy(X_test, y_test), 4))

    # Detailed report
    y_pred = mlp.predict(X_test)
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

    # Save weights and scaler
    np.savez("mlp_weights.npz",
             **{f"W{i}": p["W"] for i, p in enumerate(mlp.params)},
             **{f"b{i}": p["b"] for i, p in enumerate(mlp.params)})
    joblib.dump(scaler, "scaler.joblib")
    print("Weights saved in mlp_weights.npz and scaler in scaler.joblib")

    # Example of loading/inference
    mlp2, sc2 = load_model("mlp_weights.npz", "scaler.joblib", arch=(2,128,64,2))
    probs = predict_proba(mlp2, sc2, [[0.3, -1.2], [1.0, 0.1]])
    print("Example probabilities (2 samples):\n", probs)
    print("== Training end ==")

if __name__ == "__main__":
    run_demo()
