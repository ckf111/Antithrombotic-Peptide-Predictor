import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cv_summary(results, save_path):
    folds = [f"Fold {i+1}" for i in range(len(results))]
    metrics = ["AUC", "Accuracy", "Recall", "F1-Score"]
    
    # data[metric][fold]
    data = {
        "AUC": [r['auc'] for r in results],
        "Accuracy": [r['acc'] for r in results],
        "Recall": [r['recall'] for r in results],
        "F1-Score": [r['f1'] for r in results]
    }
    
    x = np.arange(len(folds))
    width = 0.2
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    
    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, [round(m, 4) for m in measurement], width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=8)
        multiplier += 1
        
    ax.set_ylabel('Score')
    ax.set_title('5-Fold Cross Validation Performance Summary')
    ax.set_xticks(x + width * 1.5, folds)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0.8, 1.05)
    
    plt.savefig(save_path)
    print(f"CV summary plot saved to {save_path}")

if __name__ == "__main__":
    # Results from terminal output
    cv_results = [
        {'auc': 0.9919, 'acc': 0.9735, 'recall': 0.9608, 'f1': 0.9608},
        {'auc': 0.9924, 'acc': 0.9668, 'recall': 0.9802, 'f1': 0.9519},
        {'auc': 0.9973, 'acc': 0.9801, 'recall': 0.9802, 'f1': 0.9706},
        {'auc': 0.9675, 'acc': 0.9601, 'recall': 0.9901, 'f1': 0.9434},
        {'auc': 0.9575, 'acc': 0.9435, 'recall': 0.9604, 'f1': 0.9194}
    ]
    
    out_dir = "/home/SCS2026004/ckf_workspace2/code/results"
    os.makedirs(out_dir, exist_ok=True)
    plot_cv_summary(cv_results, os.path.join(out_dir, "cv_performance_summary.png"))
