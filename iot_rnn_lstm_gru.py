"""
Comparative study: SimpleRNN vs LSTM vs GRU on three IoT applications.

  A. Smart-meter energy forecasting   (UCI Individual Household Electric Power Consumption)
  B. Weather-station temperature      (Daily minimum temperatures, Melbourne 1981-1990)
  C. Smart-building occupancy         (UCI Occupancy Detection: temp, humidity, CO2 sensors)
  D. Long-window stress test          (task A with a 168-hour / one-week look-back)

Every model gets the same budget: one recurrent layer of 64 units + a Dense head,
Adam (lr 1e-3), batch 64, up to 40 epochs, early stopping (patience 5) on validation loss.
Each configuration is trained with 3 random seeds; results are reported as mean +/- std.

Run:  KERAS_BACKEND=jax python3 iot_rnn_lstm_gru.py      (also works with tensorflow / torch backends)
Data: put the three datasets in ./data (download links in the report).
"""
import os, time, json, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
UNITS, EPOCHS, BATCH, PATIENCE = 64, 40, 64, 5
CELLS = {"RNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}


# ----------------------------------------------------------------------------- data
def windows(X, y, lookback, horizon=1):
    """Sliding windows: X[t-lookback:t] -> y[t+horizon-1]."""
    idx = np.arange(lookback, len(X) - horizon + 1)
    Xw = np.stack([X[i - lookback:i] for i in idx]).astype("float32")
    yw = y[idx + horizon - 1].astype("float32")
    return Xw, yw


def chrono_split(n, a=0.70, b=0.85):
    return slice(0, int(n * a)), slice(int(n * a), int(n * b)), slice(int(n * b), n)


def load_energy(lookback):
    df = pd.read_csv(os.path.join(DATA, "household_power_consumption.txt"), sep=";",
                     na_values="?", low_memory=False)
    df.index = pd.to_datetime(df.Date + " " + df.Time, format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).astype("float32")
    df = df.resample("1h").mean().interpolate(limit_direction="both")     # hourly smart-meter feed
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    tr, va, te = chrono_split(len(df))
    sx = StandardScaler().fit(df.iloc[tr]); sy = StandardScaler().fit(df.iloc[tr][["Global_active_power"]])
    X = sx.transform(df); y = sy.transform(df[["Global_active_power"]]).ravel()
    out = {}
    for name, s in zip(["train", "val", "test"], [tr, va, te]):
        out[name] = windows(X[s], y[s], lookback)
    return out, sy, "regression", "kW"


def load_temperature(lookback=30):
    df = pd.read_csv(os.path.join(DATA, "daily-min-temperatures.csv"))
    v = df.Temp.values.reshape(-1, 1).astype("float32")
    tr, va, te = chrono_split(len(v))
    s = StandardScaler().fit(v[tr]); z = s.transform(v)
    out = {n: windows(z[sl], z[sl].ravel(), lookback) for n, sl in zip(["train", "val", "test"], [tr, va, te])}
    return out, s, "regression", "degC"


def load_occupancy(lookback=30):
    cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]   # all five room sensors
    frames = {n: pd.read_csv(os.path.join(DATA, f)) for n, f in
              [("train", "datatraining.txt"), ("val", "datatest.txt"), ("test", "datatest2.txt")]}
    s = StandardScaler().fit(frames["train"][cols])
    out = {}
    for n, f in frames.items():
        X = s.transform(f[cols]); X = np.hstack([X, np.vstack([np.zeros((1, len(cols))), np.diff(X, axis=0)])])
        out[n] = windows(X, f.Occupancy.values, lookback)
    return out, None, "classification", ""


# ----------------------------------------------------------------------------- model
def build(cell, input_shape, task):
    m = keras.Sequential([keras.Input(input_shape), CELLS[cell](UNITS),
                          layers.Dense(1, activation="sigmoid" if task == "classification" else None)])
    m.compile(keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy" if task == "classification" else "mse")
    return m


def macs_per_inference(cell, n_in, steps):
    gates = {"RNN": 1, "LSTM": 4, "GRU": 3}[cell]
    return steps * gates * UNITS * (n_in + UNITS) + UNITS


def latency_ms(model, x1, reps=200):
    f = model.predict_on_batch
    for _ in range(20): f(x1)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(x1); t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def sigmoid(x): return 1 / (1 + np.exp(-x))


def numpy_forward(model, cell, x):
    """Plain-NumPy forward pass of the trained network for ONE window (T, F) —
    the same step-by-step loop an edge gateway / microcontroller would run."""
    rec, dense = model.layers[0], model.layers[1]
    W = [np.asarray(w) for w in rec.get_weights()]
    h = np.zeros(UNITS, dtype="float32"); c = np.zeros(UNITS, dtype="float32")
    for xt in x:
        if cell == "RNN":
            h = np.tanh(xt @ W[0] + h @ W[1] + W[2])
        elif cell == "LSTM":
            z = xt @ W[0] + h @ W[1] + W[2]
            i, f, g, o = np.split(z, 4)
            c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h = sigmoid(o) * np.tanh(c)
        else:  # GRU, Keras default reset_after=True, gate order (z, r, h)
            xz, xr, xh = np.split(xt @ W[0] + W[2][0], 3)
            hz, hr, hh = np.split(h @ W[1] + W[2][1], 3)
            zg, rg = sigmoid(xz + hz), sigmoid(xr + hr)
            hcand = np.tanh(xh + rg * hh)
            h = zg * h + (1 - zg) * hcand
    Wd, bd = [np.asarray(w) for w in dense.get_weights()]
    out = h @ Wd + bd
    return sigmoid(out) if dense.activation.__name__ == "sigmoid" else out


def numpy_latency_ms(model, cell, x, reps=300):
    for _ in range(10): numpy_forward(model, cell, x)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); numpy_forward(model, cell, x); t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def run(app, data, scaler, task, unit, cell, seed):
    keras.utils.set_random_seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    m = build(cell, Xtr.shape[1:], task)
    es = keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    t0 = time.perf_counter()
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
              callbacks=[es], verbose=0, shuffle=True)
    train_s = time.perf_counter() - t0
    p = m.predict(Xte, batch_size=1024, verbose=0).ravel()
    # the NumPy re-implementation must reproduce Keras exactly, otherwise its timing means nothing
    assert np.allclose(numpy_forward(m, cell, Xte[0]).ravel(), p[:1], atol=1e-4), "numpy forward mismatch"
    r = dict(app=app, model=cell, seed=seed, params=m.count_params(),
             epochs=len(h.history["loss"]), best_epoch=int(np.argmin(h.history["val_loss"])) + 1,
             train_s=train_s, s_per_epoch=train_s / len(h.history["loss"]),
             latency_ms=latency_ms(m, Xte[:1]),
             numpy_latency_ms=numpy_latency_ms(m, cell, Xte[0]),
             macs=macs_per_inference(cell, Xtr.shape[2], Xtr.shape[1]),
             size_kb=m.count_params() * 4 / 1024, n_train=len(Xtr), n_test=len(Xte),
             lookback=Xtr.shape[1], n_features=Xtr.shape[2], unit=unit)
    if task == "regression":
        yt = scaler.inverse_transform(yte.reshape(-1, 1)).ravel()
        yp = scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        r.update(rmse=float(np.sqrt(mean_squared_error(yt, yp))), mae=float(mean_absolute_error(yt, yp)),
                 r2=float(r2_score(yt, yp)))
        # persistence baseline: "next value = last observed value" (feature 0 is the target signal)
        naive = scaler.inverse_transform(Xte[:, -1, 0].reshape(-1, 1)).ravel()
        r["naive_rmse"] = float(np.sqrt(mean_squared_error(yt, naive)))
    else:
        yb = (p > 0.5).astype(int)
        r.update(accuracy=float(accuracy_score(yte, yb)), f1=float(f1_score(yte, yb)))
    r["val_curve"] = [float(v) for v in h.history["val_loss"]]
    return r


if __name__ == "__main__":
    apps = sys.argv[1:] or ["B", "C", "A", "D"]
    loaders = {"A": lambda: load_energy(24), "B": load_temperature, "C": load_occupancy,
               "D": lambda: load_energy(168)}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for app in apps:
        data, scaler, task, unit = loaders[app]()
        print(app, {k: v[0].shape for k, v in data.items()}, flush=True)
        for cell in CELLS:
            for seed in SEEDS:
                r = run(app, data, scaler, task, unit, cell, seed)
                with open(out_path, "a") as f: f.write(json.dumps(r) + "\n")
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "val_curve"}, flush=True)
