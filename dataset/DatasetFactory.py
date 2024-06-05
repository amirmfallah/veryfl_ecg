import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import logging

logger = logging.getLogger(__name__)

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        # Convert all object type columns to categorical codes initially
        for col in self.data_frame.columns:
            if self.data_frame[col].dtype == 'object':
                self.data_frame[col] = self.data_frame[col].astype('category').cat.codes
        self.transform = transform
        print(self.data_frame.head())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Retrieve row by index
        row = self.data_frame.iloc[idx]
        
        # Splitting the features and label
        # Assuming the label is the last column
        features = row[:-1].values.astype(float)  # Convert features to float
        label = row[-1].astype(int)


        # If there's any transformation to apply
        if self.transform:
            features = self.transform(features)
        
        # Converting to tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        return features, label
    
class DatasetFactory:
    def __init__(self, data_dir)->None:
        self.data_dir = data_dir

    def get_dataset(self, dataset_index:int, transform=None)->Dataset:
        """
        Supports datasets from data1.csv to data9.csv stored in a specified directory.
        """
        if 1 <= dataset_index <= 15:
            csv_file = os.path.join(self.data_dir, f'data{dataset_index}.csv')
            return CSVDataset(csv_file, transform)
        else:
            logger.error("Dataset index %s is out of the allowed range", dataset_index)
            raise Exception(f"Dataset index {dataset_index} is out of the allowed range")
