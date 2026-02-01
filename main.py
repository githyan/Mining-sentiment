import os

# Forced to use HuggingFace transformers model in Local
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.model import initialize_model
from src.preprocessing import drop_emptylist
from src.CustomDataset import CustomSentimentDataset
from src.trains_eval import train 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)
    torch.set_num_threads(8)


def splits():
    print("Hello from project-skripsi!")
    dataframe = pd.read_csv('vader_dataset_sentiment.csv')
    dataframe_resampled = drop_emptylist(dataframe)

    # Step 1: Split data into training (70%) and temporary (30%) sets
    X = dataframe_resampled['cleaned_comments']
    y = dataframe_resampled['sentiments']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 2: Split the temporary set (30%) into validation (15%) and test (15%) sets
    # Since X_temp is 30% of the original data, 0.5 of X_temp will be 15% of the original data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )   

    print(f'X_train shape: {X_train.shape} y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape} y_test shape: {y_test.shape}')
    print(f'X_val shape: {X_val.shape} y_val shape: {y_val.shape}')
    # Create a mapping from sentiment labels to numerical values
    label_mapping = {'positive': 1, 'neutral': 0, 'negative': 2}

    # Apply the mapping to the sentiment labels
    y_train_numerical = y_train.map(label_mapping).values
    y_val_numerical = y_val.map(label_mapping).values
    y_test_numerical = y_test.map(label_mapping).values

    print("Numerical labels created successfully:")
    print(f"y_train_numerical shape: {y_train_numerical.shape}")
    print(f"y_val_numerical shape: {y_val_numerical.shape}")
    print(f"y_test_numerical shape: {y_test_numerical.shape}")
    print("Example of y_train_numerical:", y_train_numerical[:5])

    return X_train, X_val, X_test, y_train_numerical, y_val_numerical, y_test_numerical
if __name__ == "__main__":
    set_seed(42)
    BATCH_SIZE=16
    X_train, X_val, X_test, Y_train_nums, Y_val_nums, Y_test_nums = splits()
    bert, model, optimizer, device, tokenizer = initialize_model()

    train_dataset = CustomSentimentDataset(X_train, Y_train_nums, tokenizer)
    val_dataset = CustomSentimentDataset(X_val, Y_val_nums, tokenizer)
    test_dataset = CustomSentimentDataset(X_test, Y_test_nums, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(tokenizer)
    print(train_dataset[0])
    print("Cek Parameter BERT: ", any(p.requires_grad for p in bert.parameters()))
    print("Cek Parameter model: ", any(p.requires_grad for p in model.parameters()))
    train(bert, model, optimizer, train_dataloader, val_dataloader=None)
