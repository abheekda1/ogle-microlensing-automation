## Finding clusters within gravitational lensing data

### Usage
First, clone the repo and switch to the `dev` branch:
```sh
git clone https://github.com/abheekda1/ogle-microlensing-automation
git checkout dev
```

Then, install dependencies:
```sh
pip install -r requirements.txt
```

Data should already be downloaded, but feel free to take a look at `notebooks/00_data_download.ipynb`.

Finally, run all cells of `notebooks/01_feature_engineering.ipynb` to see UMAP colored by clusters, and example plots within each cluster.
