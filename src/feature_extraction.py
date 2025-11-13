import numpy as np
from scipy import stats, signal


def _safe_div(num, denom):
    if denom is None:
        return np.nan
    if not np.isfinite(denom) or np.abs(denom) < 1e-12:
        return np.nan
    return num / denom


def _duration_above_level(flux, t, level_flux):
    mask = flux >= level_flux
    return np.ptp(t[mask]) if np.any(mask) else np.nan


def _stetson_j(flux, err):
    if err is None:
        return np.nan
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if good.sum() < 3:
        return np.nan
    flux = flux[good]
    err = err[good]
    w = 1.0 / (err ** 2)
    mean = np.sum(w * flux) / np.sum(w)
    n = len(flux)
    delta = np.sqrt(n / (n - 1)) * (flux - mean) / err
    prod = delta[:-1] * delta[1:]
    if len(prod) == 0:
        return np.nan
    return np.sum(np.sign(prod) * np.sqrt(np.abs(prod))) / len(prod)


def extract_simple_features(t, flux, err=None, seeing=None, bg=None):
    import numpy as np
    from scipy import stats

    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None
    seeing = np.asarray(seeing, dtype=float) if seeing is not None else None
    bg = np.asarray(bg, dtype=float) if bg is not None else None

    f = {}
    f["n_points"] = len(flux)
    f["duration"] = np.ptp(t) if len(t) > 0 else np.nan
    if len(t) > 1:
        cadence = np.diff(t)
        f["cadence_med"] = np.median(cadence)
        f["cadence_std"] = np.std(cadence)
        f["cadence_iqr"] = np.subtract(*np.percentile(cadence, [75, 25]))
        f["cadence_min"] = np.min(cadence)
        f["cadence_max"] = np.max(cadence)
        med_cadence = np.median(cadence)
        f["cadence_max_gap"] = np.max(cadence)
        f["cadence_frac_large_gaps"] = (
            np.mean(cadence > 3 * med_cadence) if med_cadence > 0 else np.nan
        )
    else:
        f["cadence_med"] = np.nan
        f["cadence_std"] = np.nan
        f["cadence_iqr"] = np.nan
        f["cadence_min"] = np.nan
        f["cadence_max"] = np.nan
        f["cadence_max_gap"] = np.nan
        f["cadence_frac_large_gaps"] = np.nan

    f["flux_mean"] = np.mean(flux)
    f["flux_median"] = np.median(flux)
    if len(flux) > 0:
        quantiles = np.quantile(flux, [0.05, 0.25, 0.75, 0.95])
    else:
        quantiles = np.array([np.nan, np.nan, np.nan, np.nan])
    f["flux_p05"], f["flux_q25"], f["flux_q75"], f["flux_p95"] = quantiles
    f["flux_iqr"] = f["flux_q75"] - f["flux_q25"]
    f["flux_p95_p05"] = f["flux_p95"] - f["flux_p05"]
    f["flux_mad"] = np.median(np.abs(flux - f["flux_median"]))
    f["flux_std"] = np.std(flux)
    f["flux_amp"] = np.ptp(flux)
    f["flux_skew"] = stats.skew(flux)
    f["flux_kurt"] = stats.kurtosis(flux)
    f["amp_norm"] = _safe_div(f["flux_amp"], f["flux_mean"])
    f["std_norm"] = _safe_div(f["flux_std"], f["flux_mean"])
    f["amp_over_std"] = _safe_div(f["flux_amp"], f["flux_std"])
    f["amp_over_iqr"] = _safe_div(f["flux_amp"], f["flux_iqr"])
    f["std_over_mad"] = _safe_div(f["flux_std"], f["flux_mad"])

    # symmetry
    idx_peak = np.argmax(flux) if len(flux) else 0
    t_peak = t[idx_peak] if len(t) else np.nan
    left = flux[t < t_peak]
    right = flux[t > t_peak]
    if len(left) > 3 and len(right) > 3:
        f["flux_asym"] = np.mean(left) - np.mean(right)
        f["rise_fall_ratio"] = (t_peak - t[0]) / (t[-1] - t_peak)
        pre_mean = np.mean(left)
        post_mean = np.mean(right)
        f["pre_peak_mean"] = pre_mean
        f["post_peak_mean"] = post_mean
        f["pre_post_ratio"] = _safe_div(pre_mean, post_mean)
        f["pre_post_median_diff"] = np.median(left) - np.median(right)
    else:
        f["flux_asym"] = np.nan
        f["rise_fall_ratio"] = np.nan
        f["pre_peak_mean"] = np.nan
        f["post_peak_mean"] = np.nan
        f["pre_post_ratio"] = np.nan
        f["pre_post_median_diff"] = np.nan

    if len(t) > 0:
        duration = t[-1] - t[0]
        rise_time = t_peak - t[0] if len(t) else np.nan
        fall_time = t[-1] - t_peak if len(t) else np.nan
        f["rise_rate"] = (
            (flux[idx_peak] - flux[0]) / rise_time if rise_time and rise_time > 0 else np.nan
        )
        f["fall_rate"] = (
            (flux[-1] - flux[idx_peak]) / fall_time if fall_time and fall_time > 0 else np.nan
        )
        f["peak_time_fraction"] = _safe_div(rise_time, duration)
    else:
        f["rise_rate"] = np.nan
        f["fall_rate"] = np.nan
        f["peak_time_fraction"] = np.nan

    # width
    min_flux = np.min(flux)
    half = min_flux + 0.5 * f["flux_amp"]
    mask = flux >= half
    f["fwhm_time"] = np.ptp(t[mask]) if np.any(mask) else np.nan
    quarter = min_flux + 0.25 * f["flux_amp"]
    three_quarter = min_flux + 0.75 * f["flux_amp"]
    f["t_above_q25"] = _duration_above_level(flux, t, quarter)
    f["t_above_q75"] = _duration_above_level(flux, t, three_quarter)

    # quality
    resid = flux - np.median(flux)
    sigma = np.std(resid)
    f["outlier_frac"] = np.mean(np.abs(resid) > 3 * sigma) if sigma > 0 else 0.0
    f["frac_above_1sigma"] = (
        np.mean(resid > sigma) if sigma > 0 else np.nan
    )
    f["frac_above_2sigma"] = (
        np.mean(resid > 2 * sigma) if sigma > 0 else np.nan
    )
    f["von_neumann_ratio"] = (
        np.sum(np.diff(flux) ** 2) / ((len(flux) - 1) * np.var(flux, ddof=1))
        if len(flux) > 2 and np.var(flux, ddof=1) > 0
        else np.nan
    )
    if len(flux) > 2:
        num = np.corrcoef(flux[:-1], flux[1:])
        f["autocorr_lag1"] = num[0, 1] if np.isfinite(num[0, 1]) else np.nan
    else:
        f["autocorr_lag1"] = np.nan

    if err is not None and len(err) == len(flux):
        err_flux = np.abs(0.4 * np.log(10) * flux * err)
        valid = np.isfinite(err_flux) & (err_flux > 0)
        f["err_flux_median"] = np.median(err_flux[valid]) if np.any(valid) else np.nan
        f["err_flux_mean"] = np.mean(err_flux[valid]) if np.any(valid) else np.nan
        snr = np.divide(
            flux,
            err_flux,
            out=np.full_like(flux, np.nan, dtype=float),
            where=err_flux > 0,
        )
        f["snr_median"] = np.nanmedian(snr) if np.any(np.isfinite(snr)) else np.nan
        f["snr_peak"] = snr[idx_peak] if idx_peak < len(snr) else np.nan
        if np.any(valid):
            flux_valid = flux[valid]
            err_valid = err_flux[valid]
            mean_flux = np.mean(flux_valid)
            chi2 = np.sum(((flux_valid - mean_flux) / err_valid) ** 2)
            dof = len(flux_valid) - 1
            f["chi2_const"] = chi2
            f["chi2_reduced"] = chi2 / dof if dof > 0 else np.nan
        else:
            f["chi2_const"] = np.nan
            f["chi2_reduced"] = np.nan
        f["stetson_j"] = _stetson_j(flux, err_flux)
    else:
        f["err_flux_median"] = np.nan
        f["err_flux_mean"] = np.nan
        f["snr_median"] = np.nan
        f["snr_peak"] = np.nan
        f["chi2_const"] = np.nan
        f["chi2_reduced"] = np.nan
        f["stetson_j"] = np.nan

    if seeing is not None and len(seeing) == len(flux):
        f["seeing_mean"] = np.mean(seeing)
        f["seeing_std"] = np.std(seeing)
        if np.std(seeing) > 0 and np.std(flux) > 0:
            corr = np.corrcoef(seeing, flux)[0, 1]
            f["seeing_flux_corr"] = corr if np.isfinite(corr) else np.nan
            cov = np.cov(seeing, flux, ddof=1)
            f["seeing_flux_slope"] = cov[0, 1] / np.var(seeing, ddof=1)
        else:
            f["seeing_flux_corr"] = np.nan
            f["seeing_flux_slope"] = np.nan
    else:
        f["seeing_mean"] = np.mean(seeing) if seeing is not None else np.nan
        f["seeing_std"] = np.std(seeing) if seeing is not None else np.nan
        f["seeing_flux_corr"] = np.nan
        f["seeing_flux_slope"] = np.nan

    if bg is not None and len(bg) == len(flux):
        f["bg_mean"] = np.mean(bg)
        f["bg_std"] = np.std(bg)
        if np.std(bg) > 0 and np.std(flux) > 0:
            corr = np.corrcoef(bg, flux)[0, 1]
            f["bg_flux_corr"] = corr if np.isfinite(corr) else np.nan
            cov = np.cov(bg, flux, ddof=1)
            f["bg_flux_slope"] = cov[0, 1] / np.var(bg, ddof=1)
        else:
            f["bg_flux_corr"] = np.nan
            f["bg_flux_slope"] = np.nan
    else:
        f["bg_mean"] = np.mean(bg) if bg is not None else np.nan
        f["bg_std"] = np.std(bg) if bg is not None else np.nan
        f["bg_flux_corr"] = np.nan
        f["bg_flux_slope"] = np.nan

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
