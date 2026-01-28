# Grables: Tabular Learning Beyond Independent Rows


This repository contains the code used to generate the experimental results for our paper.

The full, refactored source code will be released upon publication of the paper.

The code provided here is organized as self-contained Jupyter notebooks, grouped by dataset

---

## Repository Structure

```
notebooks/
  synthetic/
    00_create_data_and_tasks.ipynb
    10_train_tabular.ipynb
    20_train_tab_gnn.ipynb

  retail/
    00_prepare_data_and_tasks.ipynb
    10_train_tabular.ipynb
    20_train_tab_gnn.ipynb

  relbench_trial/
    00_prepare_data.ipynb
    10_tabular.ipynb          # includes NFA
    20_tab_gnn.ipynb
    30_tab_gnn_db.ipynb

data/
  raw/        # datasets downloaded by the user
  processed/  # generated splits and features

runs/
  ...         # model outputs, metrics, checkpoints
```

Each dataset has its own folder, and notebooks are numbered to indicate the suggested **execution order**.

---

## Environment Setup

### Python
The code was run with **Python 3.9+**.

### Dependencies
Install dependencies using:

```bash
pip install -r requirements.txt
```

**Note:**
- Installing `torch`, `torch_geometric`, and `autogluon` may require following their official installation instructions depending on your CUDA / OS setup.
- If PyTorch Geometric wheels are not found automatically, see:
  https://pytorch-geometric.readthedocs.io

---

## Dataset Instructions

### 1. Synthetic Data
No external data is required. Notebook `notebooks/synthetic/00_create_data_and_tasks.ipynb` creates the data and tasks. 

---

### 2. Retail Dataset
The experiments use the **UCI Online Retail dataset**.

1. Download the dataset from:
   https://archive.ics.uci.edu/ml/datasets/online+retail
2. Place the file at:
   `data/raw/Online Retail.xlsx`

You should run `notebooks/retail/00_prepare_data_and_tasks.ipynb` to instantiate the logical tasks. 

---

### 3. RelBench Trial Dataset
These experiments use the **RelBench** benchmark.

The notebooks automatically download the required data via the `relbench` library.

---

## Reproducing Results

To reproduce the paper results:

1. Set up the environment.
2. Download required datasets.
3. Recommended to run the notebooks **in numerical order** within each dataset folder.

---

## Known Limitations

- Exact runtime may vary depending on hardware (CPU/GPU).
- Some models (e.g., AutoGluon) may use nondeterministic components despite fixed seeds.
- GPU memory requirements vary for GNN experiments.

---

## License

This code is released for research and reproducibility purposes.