
# ASTR 356 — Final Project: UMAP & Clustering with OGLE Microlensing Data to Automate Detection of Anomalies

This repository contains code, notebooks, and data used for Diego Miura and Abheek Dhawan's ASTR 356 final project. The project extracts time-series features from OGLE photometry, performs dimensionality reduction with UMAP, and explores clustering to classify gravitational lensing events.

**Key Points**
- **Data:** OGLE photometry and derived features (see `data/` and `data/features/`).
- **Notebooks:** Analysis is organized as reproducible Jupyter notebooks in `notebooks/`.
- **Code:** Reusable helper functions and feature-extraction logic live in `src/`.
- **Results:** Figures and derived outputs are saved under `results/` and `figures/`.

**Repository Structure**
- **Notebooks:** [notebooks/00_data_download.ipynb](notebooks/00_data_download.ipynb), [notebooks/01_feature_engineering.ipynb](notebooks/01_feature_engineering.ipynb), [notebooks/02_umap_and_clustering.ipynb](notebooks/02_umap_and_clustering.ipynb)
- **Source:** [src/feature_extraction.py](src/feature_extraction.py), [src/preprocessing.py](src/preprocessing.py), [src/util.py](src/util.py)
- **Data:** [data/raw/ogle4/2025](data/raw/ogle4/2025) (raw photometry), [data/features/features_ogle2025.csv](data/features/features_ogle2025.csv)
- **Results & figures:** [results/](results/) and [figures/](figures/)

**Quickstart**

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -n astr356 python=3.11
conda activate astr356
pip install -r requirements.txt
```

2. Run the notebooks in order to reproduce the analysis and figures:

- `notebooks/00_data_download.ipynb` — obtain and organize raw OGLE data (if needed).
- `notebooks/01_feature_engineering.ipynb` — compute or load time-series features and preprocessing.
- `notebooks/02_umap_and_clustering.ipynb` — run UMAP, clustering, and save summary figures.

Start Jupyter Lab or Notebook and run the cells sequentially:

```bash
jupyter lab
```

**Scripts & Modules**
- Use functions in `src/feature_extraction.py` and `src/preprocessing.py` from the notebooks. The notebooks show example calls and expected inputs/outputs.

**Data Notes**
- The repository includes feature tables in `data/features/`:
	- `features_ogle2025.csv` — full feature set used for analysis.
	- `features_ogle2025_simple.csv` — older, reduced feature subset.
- Raw photometry (subset) and metadata are under `data/raw/ogle4/2025/`.

**Reproducing Figures**
- Run the notebooks end-to-end and export figures (notebook cells save output images to `results/figures/`). See the plotting cells in `notebooks/02_umap_and_clustering.ipynb`.

**Dependencies**
- See `requirements.txt` for the Python package dependencies.

**License & Attribution**
- This project is released under the terms in the `LICENSE` file.

**Contact**
- For questions or changes, open an issue or contact the author.
