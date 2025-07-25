import pandas as pd, numpy as np, pickle, pathlib as pl

root = pl.Path(__file__).resolve().parent.parent.parent  # adjust if running elsewhere
raw   = pd.read_parquet(root / "processed_data" / "calendar.parquet")
scaled= pd.read_parquet(root / "processed_data" / "calendar_scaled.parquet")

print("RAW shape   :", raw.shape)
print("SCALED shape:", scaled.shape)

print("Dates:", raw['date'].min().date(), "→", raw['date'].max().date())
print("Missing values per column (raw):\n", raw.isna().sum().sort_values().tail(10))

scaled_cols = [c for c in scaled.columns if c.endswith("_scaled")]
stats = scaled[scaled_cols].agg(["min", "max"]).T

print(stats.head())
assert np.allclose(stats["min"].min(), 0, atol=1e-6)
assert np.allclose(stats["max"].max(), 1, atol=1e-6)

print("\nRAW tail:")
print(raw.tail()[["date"] + raw.columns.tolist()[1:6]])   # first few cols

print("\nSCALED tail:")
print(scaled.tail()[["date"] + scaled_cols[:5]])

with open(root / "processed_data" / "feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# pick one row
row = scaled.loc[scaled.index[100], scaled_cols].values.reshape(1, -1)
orig = scaler.inverse_transform(row)

print("Scaled → original (first 5 features):")
print(orig[0, :5])

import matplotlib.pyplot as plt
plt.plot(raw['date'][-365:], raw['ethanol'][-365:], label="ethanol raw")
plt.plot(scaled['date'][-365:], scaled['ethanol_scaled'][-365:] * (raw['ethanol'].max()-raw['ethanol'].min()) + raw['ethanol'].min(),
         label="rescaled back")
plt.legend(); plt.title("Visual cross‑check")
plt.show()