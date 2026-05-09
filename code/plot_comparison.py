
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison():
    results_dir = "/home/SCS2026004/ckf_workspace2/code/results"
    
    # Load results
    hybrid_df = pd.read_csv(os.path.join(results_dir, "hybrid_results.csv"))
    traditional_df = pd.read_csv(os.path.join(results_dir, "traditional_results.csv"))
    
    # Combine
    all_df = pd.concat([hybrid_df, traditional_df], ignore_index=True)
    
    # Metrics to plot
    metrics = ["auc", "acc", "recall", "f1", "mcc"]
    metric_labels = ["AUC", "Accuracy", "Recall", "F1-Score", "MCC"]
    
    # Set up plot
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = all_df["model"].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model_name in enumerate(models):
        model_data = all_df[all_df["model"] == model_name].iloc[0]
        values = [model_data[m] for m in metrics]
        rects = ax.bar(x + (i - 1) * width, values, width, label=model_name, color=colors[i % len(colors)])
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=9)
        
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (5-Fold CV Average)')
    ax.set_xticks(x, metric_labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0.8, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    plot_comparison()
