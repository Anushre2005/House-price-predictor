# save as src/export_california.py (optional helper)
from sklearn.datasets import fetch_california_housing
import pandas as pd, os
X, y = fetch_california_housing(as_frame=True, return_X_y=True)
df = X.copy()
df["MedHouseVal"] = y  # target
os.makedirs("data", exist_ok=True)
df.to_csv("data/train.csv", index=False)
print(df.head())
