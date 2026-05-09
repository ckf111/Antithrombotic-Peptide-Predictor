
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import numpy as np
import os
import pandas as pd
from models import HybridModel
from dataset import PeptideDataset, load_all_data

CONFIG = {
    "input_dim": 320,
    "cnn_channels": 64,
    "lstm_hidden": 128,
    "dropout": 0.3,
    "batch_size": 32,
    "save_dir": "/home/SCS2026004/ckf_workspace2/code/checkpoints"
}

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = {
        "auc": roc_auc_score(all_labels, all_probs),
        "acc": accuracy_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "mcc": matthews_corrcoef(all_labels, all_preds)
    }
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = load_all_data()
    dataset = PeptideDataset(features, labels)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        val_sub = Subset(dataset, val_idx)
        val_loader = DataLoader(val_sub, batch_size=CONFIG["batch_size"], shuffle=False)
        
        model = HybridModel(input_dim=CONFIG["input_dim"], 
                            cnn_channels=CONFIG["cnn_channels"],
                            lstm_hidden=CONFIG["lstm_hidden"],
                            dropout=CONFIG["dropout"]).to(device)
        
        checkpoint_path = os.path.join(CONFIG["save_dir"], f"best_model_fold{fold+1}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Error: {checkpoint_path} not found!")
            continue
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        metrics = evaluate(model, val_loader, device)
        cv_results.append(metrics)
        print(f"Fold {fold+1} AUC: {metrics['auc']:.4f}")

    # Summary Table
    avg_res = {k: np.mean([r[k] for r in cv_results]) for k in cv_results[0].keys()}
    avg_res["model"] = "Hybrid Model (CNN+BiLSTM+Attn)"
    
    df = pd.DataFrame([avg_res])
    print("\nHybrid Model Summary:")
    print(df[["model", "auc", "acc", "recall", "f1", "mcc"]])
    
    # Save results
    out_dir = "/home/SCS2026004/ckf_workspace2/code/results"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "hybrid_results.csv"), index=False)

if __name__ == "__main__":
    main()
