import numpy as np
import pandas as pd


def preprocess_lc(df):
    """
    Sorts a light curve, drops missing rows, converts magnitudes to relative
    flux, and adds a time-relative column. Matches the behavior used inside
    the exploratory notebooks so downstream scripts can share the logic.
    """
    if df.empty:
        return df.copy()

    cols = ["HJD", "mag", "err"]
    present = [c for c in cols if c in df.columns]
    df = df.dropna(subset=present).sort_values("HJD").reset_index(drop=True)
    if df.empty:
        return df

    n_top = max(1, int(len(df) * 0.3))
    baseline = np.median(df["mag"].nlargest(n_top))
    df = df.copy()
    df["flux"] = 10 ** (-0.4 * (df["mag"] - baseline))
    df["t_rel"] = df["HJD"] - df["HJD"].min()
    return df
