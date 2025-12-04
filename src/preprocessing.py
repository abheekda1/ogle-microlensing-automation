import numpy as np

def preprocess_lc(df):
    df = df.sort_values("HJD").dropna()
    baseline = np.median(df["mag"].nlargest(int(len(df)*0.3)))
    df["flux"] = 10**(-0.4 * (df["mag"] - baseline))
    df["t_rel"] = df["HJD"] - df["HJD"].min()
    return df
