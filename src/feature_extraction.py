import numpy as np
from scipy import stats, signal

def extract_simple_features(t, flux, err=None, seeing=None, bg=None):
    import numpy as np
    from scipy import stats

    f = {}
    f["n_points"] = len(flux)
    f["duration"] = np.ptp(t)
    f["cadence_med"] = np.median(np.diff(t)) if len(t)>1 else np.nan
    f["flux_mean"] = np.mean(flux)
    f["flux_std"] = np.std(flux)
    f["flux_amp"] = np.ptp(flux)
    f["flux_skew"] = stats.skew(flux)
    f["flux_kurt"] = stats.kurtosis(flux)
    f["amp_norm"] = f["flux_amp"] / f["flux_mean"]
    f["std_norm"] = f["flux_std"] / f["flux_mean"]

    # symmetry
    idx_peak = np.argmax(flux)
    t_peak = t[idx_peak]
    left = flux[t < t_peak]; right = flux[t > t_peak]
    if len(left)>3 and len(right)>3:
        f["flux_asym"] = np.mean(left) - np.mean(right)
        f["rise_fall_ratio"] = (t_peak - t[0]) / (t[-1] - t_peak)
    else:
        f["flux_asym"], f["rise_fall_ratio"] = np.nan, np.nan

    # width
    half = np.min(flux) + 0.5*f["flux_amp"]
    mask = flux >= half
    f["fwhm_time"] = np.ptp(t[mask]) if np.any(mask) else np.nan

    # quality
    resid = flux - np.median(flux)
    sigma = np.std(resid)
    f["outlier_frac"] = np.mean(np.abs(resid) > 3*sigma)
    if seeing is not None:
        f["seeing_mean"] = np.mean(seeing)
        f["seeing_std"] = np.std(seeing)
    if bg is not None:
        f["bg_mean"] = np.mean(bg)
        f["bg_std"] = np.std(bg)
    return f

def extract_features(t, flux, err=None, seeing=None, bg=None):
    """
    Extract robust, cadence-insensitive morphological features
    from an OGLE microlensing light curve for UMAP+HDBSCAN clustering.

    Parameters
    ----------
    t : array-like
        Time (HJD or relative days)
    flux : array-like
        Flux or magnitude (if mag, invert sign before using)
    err : array-like, optional
        Photometric uncertainties
    seeing, bg : array-like, optional
        Seeing and background columns (if present)

    Returns
    -------
    f : dict
        Dictionary of extracted features
    """
    f = {}
    n = len(flux)
    if n < 5:
        # too few points → fill with NaNs
        return {key: np.nan for key in [
            "n_points","duration","cadence_med","flux_mean","flux_std","flux_amp",
            "flux_skew","flux_kurt","amp_norm","std_norm","flux_asym","rise_fall_ratio",
            "fwhm_time","abbe_value","outlier_frac","smoothness","seeing_mean","seeing_std",
            "bg_mean","bg_std","ls_top1","ls_top2","ls_top3"
        ]}

    # --- Basic statistics ---
    f["n_points"] = n
    f["duration"] = np.ptp(t)
    cadence = np.diff(np.sort(t))
    f["cadence_med"] = np.median(cadence) if len(cadence)>0 else np.nan
    f["flux_mean"] = np.mean(flux)
    f["flux_std"] = np.std(flux)
    f["flux_amp"] = np.ptp(flux)
    f["flux_skew"] = stats.skew(flux)
    f["flux_kurt"] = stats.kurtosis(flux)
    f["amp_norm"] = f["flux_amp"] / (f["flux_mean"] + 1e-8)
    f["std_norm"] = f["flux_std"] / (f["flux_mean"] + 1e-8)

    # --- Symmetry & shape ---
    idx_peak = np.argmax(flux)
    t_peak = t[idx_peak]
    left = flux[t < t_peak]; right = flux[t > t_peak]
    if len(left)>3 and len(right)>3:
        f["flux_asym"] = np.mean(left) - np.mean(right)
        f["rise_fall_ratio"] = (t_peak - t[0]) / (t[-1] - t_peak + 1e-8)
    else:
        f["flux_asym"], f["rise_fall_ratio"] = np.nan, np.nan

    # --- Width (FWHM) ---
    half = np.min(flux) + 0.5*f["flux_amp"]
    mask = flux >= half
    f["fwhm_time"] = np.ptp(t[mask]) if np.any(mask) else np.nan

    # --- Abbe value (Mowlavi 2014): smoothness measure ---
    diff = np.diff(flux)
    if len(diff) > 1:
        f["abbe_value"] = np.sum(diff**2) / (2 * np.sum((flux - np.mean(flux))**2))
    else:
        f["abbe_value"] = np.nan

    # --- Smoothness / even statistics ---
    # even-deviation (Ferreira Lopes & Cross 2017)
    sorted_flux = np.sort(flux)
    mid = n // 2
    if n > 1:
        even_median = 0.5 * (sorted_flux[mid] + sorted_flux[mid-1]) if n % 2 == 0 else sorted_flux[mid]
        f["smoothness"] = np.mean(np.abs(flux - even_median))
    else:
        f["smoothness"] = np.nan

    # --- Quality / outliers ---
    resid = flux - np.median(flux)
    sigma = np.std(resid)
    f["outlier_frac"] = np.mean(np.abs(resid) > 3*sigma)
    if seeing is not None:
        f["seeing_mean"] = np.mean(seeing)
        f["seeing_std"] = np.std(seeing)
    if bg is not None:
        f["bg_mean"] = np.mean(bg)
        f["bg_std"] = np.std(bg)

    # --- Simple frequency snapshot (Pantoja et al. 2022 idea) ---
    try:
        freq, power = signal.periodogram(flux - np.mean(flux), fs=1/np.median(cadence))
        top = np.sort(power)[::-1][:3] if len(power) >= 3 else [np.nan]*3
        f["ls_top1"], f["ls_top2"], f["ls_top3"] = top[0], top[1], top[2]
    except Exception:
        f["ls_top1"], f["ls_top2"], f["ls_top3"] = np.nan, np.nan, np.nan

    return f



# def extract_features(t_rel, flux, err):
#     features = {}
#     features['mean_flux'] = np.mean(flux)
#     features['std_flux'] = np.std(flux)
#     features['max_flux'] = np.max(flux)
#     features['min_flux'] = np.min(flux)
#     features['amplitude'] = features['max_flux'] - features['min_flux']
#     features['median_flux'] = np.median(flux)
#     features['flux_skewness'] = scipy.stats.skew(flux)
#     features['flux_kurtosis'] = scipy.stats.kurtosis(flux)
#     features['num_points'] = len(flux) 
#     return features