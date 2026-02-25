# Probabilistic Apriori Algorithm Implementation

An expert-level implementation of the Probabilistic Apriori algorithm for mining Frequent Itemsets and Association Rules from Uncertain Transactional Databases (UTDs).

## Features
- Handles UTDs and calculates Expected Support (`E[sup(X)]`).
- Calculates Probabilistic Association Rules using Expected Confidence metrics.
- Uses `argparse` for CLI configuration.
- Modular component design (`main.py`, `apriori.py`, `data_loader.py`).
- Deterministic behavior fallback available.

## Installation

This project uses `uv` and virtual environments for dependency management.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate  # Linux/MacOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies using uv
pip install uv
uv pip install numpy pandas
```

## Dataset Requirements
Ensure the `Groceries_dataset.csv` is correctly placed in the project root directory or dynamically specify its path.

Download the dataset here: [Kaggle Groceries Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)

## Execution
Run the script using the virtual environment interpreter:

```bash
python main.py --csv Groceries_dataset.csv --min_sup_ratio 0.001 --min_conf 0.10
```

### CLI Arguments

| Argument | Default | Description |
|-----------|---------|-------------|
| `--csv` | `Groceries_dataset.csv` | Filepath to the dataset CSV file. |
| `--min_sup_ratio` | `0.001` (0.1%) | The minimum probabilistic support ratio threshold for frequent itemsets. |
| `--min_conf` | `0.10` (10%) | The minimum expected confidence threshold for association rules. |
| `--deterministic` | `False` | Disables uncertainty simulation and processes items with `prob=1.0`. |
