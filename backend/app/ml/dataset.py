import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("phishing_platform")

def prepare_dataset(raw_csv_path: str = "spam.csv", output_dir: str = "datasets"):
    """Loads, cleans, splits and saves spam dataset."""
    if not os.path.exists(raw_csv_path):
        # Check if it's in backend or parent directory
        parent_path = os.path.join("..", raw_csv_path)
        if os.path.exists(parent_path):
            raw_csv_path = parent_path
        else:
            raise FileNotFoundError(f"Source dataset '{raw_csv_path}' not found.")

    logger.info(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    
    # Map label names to numeric representation
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label_num", "text"])
    
    # Basic text cleaning: remove extra whitespace
    df["text"] = df["text"].astype(str).str.strip()

    # Stratified splits: 80% train, 10% validation, 10% test
    # First split 80% train, 20% validation+test
    train_df, val_test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"]
    )
    
    # Split the 20% equally into validation and test (50/50 of 20% = 10% each)
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        random_state=42,
        stratify=val_test_df["label_num"]
    )

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Dataset splits saved to '{output_dir}/':")
    logger.info(f"  Train: {len(train_df)} samples (Spam: {(train_df.label_num==1).sum()})")
    logger.info(f"  Val:   {len(val_df)} samples (Spam: {(val_df.label_num==1).sum()})")
    logger.info(f"  Test:  {len(test_df)} samples (Spam: {(test_df.label_num==1).sum()})")

    return train_path, val_path, test_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_dataset()
