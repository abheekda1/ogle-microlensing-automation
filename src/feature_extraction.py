import numpy as np
import scipy.stats

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