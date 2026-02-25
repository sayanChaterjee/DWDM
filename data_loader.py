import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_and_transform_dataset(csv_path: str, simulate_uncertainty: bool = True) -> list:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'.")

    logging.info(f"Ingesting dataset from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")

    if 'Member_number' in df.columns and 'Date' in df.columns and 'itemDescription' in df.columns:
        basket_groups = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
        items_list = basket_groups['itemDescription']
    else:
        logging.warning("Standard columns not found. Attempting index-based grouping.")
        if len(df.columns) > 0:
            transaction_col = df.columns[-1]    
            items_list = [[x] for x in df[transaction_col].tolist()]
        else:
             items_list = []
    
    utd_transactions = []
    
    np.random.seed(42) 
    for items in items_list:
        transaction_dict = {}
        for item in set(items):
            if pd.isna(item):
                continue
            if simulate_uncertainty:
                prob = np.random.uniform(0.6, 1.0)
            else:
                prob = 1.0
                
            transaction_dict[item] = prob
            
        if transaction_dict:
            utd_transactions.append(transaction_dict)
        
    logging.info(f"Data Transformation Complete. Unique Transactions Compiled: {len(utd_transactions)}")
    return utd_transactions
