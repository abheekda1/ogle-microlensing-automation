import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def read_params_file(filepath):
    """Reads an OGLE params.dat file and returns a dictionary of parameters."""
    params = {
        "event": os.path.basename(filepath).replace("_params.dat", "").lower(),
        "Field": None, "StarNo": None, "RA": None, "Dec": None, "Remarks": None,
    }

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if lines:
        params["event_name"] = lines[0]

    for line in lines:
        if line.startswith("Field"):
            params["Field"] = line.split()[-1]
        elif line.startswith("StarNo"):
            params["StarNo"] = line.split()[-1]
        elif line.startswith("RA("):
            params["RA"] = line.split()[-1]
        elif line.startswith("Dec("):
            params["Dec"] = line.split()[-1]
        elif line.startswith("Remarks"):
            params["Remarks"] = " ".join(line.split()[1:]) if len(line.split()) > 1 else ""

    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0][0].isalpha():
            key, val, err = parts
            try:
                params[key] = float(val)
                params[key + "_err"] = float(err)
            except ValueError:
                pass

    return params

def pspl_model(t, t0, tE, u0, I0, fbl=1.0):
    u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    F = fbl * A + (1 - fbl)
    m = I0 - 2.5 * np.log10(F)
    return m

def category_cmap(categories, base_cmap='magma'):
    N = len(categories)
    # colors = plt.get_cmap(base).colors[:N]
    colors = plt.get_cmap(base_cmap)(np.linspace(0, 1, N))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(N+1), N)
    return cmap, norm, np.arange(N)+0.5, categories
