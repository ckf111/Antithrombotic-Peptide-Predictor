import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import pandas as pd

def load_data(features_dir="/home/SCS2026004/ckf_workspace2/code/features"):
    # Using ESM-2 features for traditional models as they are high-quality
    # We need to flatten them as traditional models expect (batch, features)
    pos_esm2 = np.load(os.path.join(features_dir, "pos_esm2.npy"))
    neg_esm2 = np.load(os.path.join(features_dir, "neg_esm2.npy"))
    
    # Global average pooling to flatten (batch, 100, 320) -> (batch, 320)
    pos_flat = pos_esm2.mean(axis=1)
    neg_flat = neg_esm2.mean(axis=1)
    
    pos_labels = np.load(os.path.join(features_dir, "pos_labels.npy"))
    neg_labels = np.load(os.path.join(features_dir, "neg_labels.npy"))
    
    X = np.concatenate([pos_flat, neg_flat], axis=0)
    y = np.concatenate([pos_labels, neg_labels], axis=0)
    
    return X, y

def run_cv_traditional(model_name, clf, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    print(f"\n--- Running 5-Fold CV for {model_name} ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf.fit(X_train, y_train)
        y_probs = clf.predict_proba(X_val)[:, 1]
        y_preds = clf.predict(X_val)
        
        metrics = {
            "auc": roc_auc_score(y_val, y_probs),
            "acc": accuracy_score(y_val, y_preds),
            "recall": recall_score(y_val, y_preds),
            "precision": precision_score(y_val, y_preds),
            "f1": f1_score(y_val, y_preds),
            "mcc": matthews_corrcoef(y_val, y_preds)
        }
        results.append(metrics)
        print(f"Fold {fold+1} AUC: {metrics['auc']:.4f}")
        
    return results

def main():
    X, y = load_data()
    
    # 1. SVM
    svm_clf = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    svm_results = run_cv_traditional("SVM", svm_clf, X, y)
    
    # 2. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_results = run_cv_traditional("Random Forest", rf_clf, X, y)
    
    # Summary Table
    models = ["SVM", "Random Forest"]
    all_results = [svm_results, rf_results]
    
    summary_data = []
    for name, res_list in zip(models, all_results):
        avg_res = {k: np.mean([r[k] for r in res_list]) for k in res_list[0].keys()}
        avg_res["model"] = name
        summary_data.append(avg_res)
        
    df = pd.DataFrame(summary_data)
    print("\nTraditional Models Summary:")
    print(df[["model", "auc", "acc", "recall", "f1", "mcc"]])
    
    # Save for later comparison
    out_dir = "/home/SCS2026004/ckf_workspace2/code/results"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "traditional_results.csv"), index=False)

if __name__ == "__main__":
    main()
