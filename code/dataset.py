import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class PeptideDataset(Dataset):
    def __init__(self, esm2_features, labels):
        self.features = torch.FloatTensor(esm2_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_all_data(features_dir="/home/SCS2026004/ckf_workspace2/code/features"):
    # Load positive and negative features
    pos_esm2 = np.load(os.path.join(features_dir, "pos_esm2.npy"))
    neg_esm2 = np.load(os.path.join(features_dir, "neg_esm2.npy"))
    
    pos_labels = np.load(os.path.join(features_dir, "pos_labels.npy"))
    neg_labels = np.load(os.path.join(features_dir, "neg_labels.npy"))
    
    # Concatenate
    all_features = np.concatenate([pos_esm2, neg_esm2], axis=0)
    all_labels = np.concatenate([pos_labels, neg_labels], axis=0)
    
    return all_features, all_labels

if __name__ == "__main__":
    # Test data loading
    features, labels = load_all_data()
    dataset = PeptideDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch_feat, batch_labels = next(iter(dataloader))
    print(f"Batch features shape: {batch_feat.shape}") # (32, 100, 320)
    print(f"Batch labels shape: {batch_labels.shape}")   # (32,)
    print("DataLoader test passed!")
