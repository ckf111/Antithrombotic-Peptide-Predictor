import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import os
import matplotlib.pyplot as plt
from models import HybridModel
from dataset import PeptideDataset, load_all_data

# Configuration
CONFIG = {
    "input_dim": 320,
    "cnn_channels": 64,
    "lstm_hidden": 128,
    "dropout": 0.3,
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10,
    "n_splits": 5,
    "save_dir": "/home/SCS2026004/ckf_workspace2/code/checkpoints"
}

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_state_dict(), path)
        self.val_loss_min = val_loss

# Correcting the typo in save_checkpoint (state_state_dict -> state_dict)
class EarlyStoppingFixed(EarlyStopping):
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        "loss": avg_loss,
        "auc": roc_auc_score(all_labels, all_preds),
        "acc": accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
        "precision": precision_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0),
        "recall": recall_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0),
        "f1": f1_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)
    }
    return metrics

def run_cv():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features, labels = load_all_data()
    dataset = PeptideDataset(features, labels)
    
    skf = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n--- Fold {fold+1}/{CONFIG['n_splits']} ---")
        
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=CONFIG["batch_size"], shuffle=False)
        
        model = HybridModel(input_dim=CONFIG["input_dim"], 
                            cnn_channels=CONFIG["cnn_channels"],
                            lstm_hidden=CONFIG["lstm_hidden"],
                            dropout=CONFIG["dropout"]).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        checkpoint_path = os.path.join(CONFIG["save_dir"], f"best_model_fold{fold+1}.pth")
        early_stopping = EarlyStoppingFixed(patience=CONFIG["patience"], verbose=True)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(CONFIG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_metrics["loss"])
            
            scheduler.step(val_metrics["loss"])
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            early_stopping(val_metrics["loss"], model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model for this fold and evaluate final metrics
        model.load_state_dict(torch.load(checkpoint_path))
        final_metrics = evaluate(model, val_loader, criterion, device)
        cv_results.append(final_metrics)
        print(f"Fold {fold+1} Final AUC: {final_metrics['auc']:.4f}")

    # Summary
    print("\n" + "="*30)
    print("Cross-Validation Summary")
    print("="*30)
    for i, res in enumerate(cv_results):
        print(f"Fold {i+1}: AUC={res['auc']:.4f}, Acc={res['acc']:.4f}, Recall={res['recall']:.4f}, F1={res['f1']:.4f}")
    
    avg_auc = np.mean([res['auc'] for res in cv_results])
    std_auc = np.std([res['auc'] for res in cv_results])
    print(f"\nAverage AUC: {avg_auc:.4f} (+/- {std_auc:.4f})")
    
    if std_auc < 0.03:
        print("✓ Stability requirement (std < 0.03) met!")
    else:
        print("✗ Stability requirement (std < 0.03) NOT met.")

if __name__ == "__main__":
    run_cv()
