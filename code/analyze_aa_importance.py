
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import HybridModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = HybridModel(input_dim=320, cnn_channels=64, lstm_hidden=128, dropout=0.3).to(device)
    checkpoint_path = "/home/SCS2026004/ckf_workspace2/code/checkpoints/best_model_fold1.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Load Data
    pos_fasta = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/positive_samples.fasta"
    with open(pos_fasta, "r") as f:
        sequences = [line.strip() for line in f if not line.startswith(">")]
    
    pos_esm2 = np.load("/home/SCS2026004/ckf_workspace2/code/features/pos_esm2.npy")
    
    # 3. Aggregate weights by amino acid type
    aa_weights = {}
    aa_counts = {}
    
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            feat = torch.FloatTensor(pos_esm2[i]).unsqueeze(0).to(device)
            _, weights = model(feat)
            weights = weights.squeeze().cpu().numpy()
            
            for j, aa in enumerate(seq):
                if aa not in aa_weights:
                    aa_weights[aa] = 0
                    aa_counts[aa] = 0
                aa_weights[aa] += weights[j]
                aa_counts[aa] += 1
                
    # 4. Calculate average weight per AA type
    avg_weights = {aa: aa_weights[aa] / aa_counts[aa] for aa in aa_weights}
    
    # 5. Plot
    df = pd.DataFrame(list(avg_weights.items()), columns=["Amino Acid", "Avg Attention Weight"])
    df = df.sort_values("Avg Attention Weight", ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Amino Acid", y="Avg Attention Weight", data=df, palette="viridis")
    plt.title("Average Attention Weight by Amino Acid Type (Positive Samples)")
    plt.axhline(y=1/100, color='r', linestyle='--', label="Uniform Average (1/100)")
    plt.legend()
    
    results_dir = "/home/SCS2026004/ckf_workspace2/code/results/interpretability"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "aa_importance.png"), dpi=300)
    print(f"Saved AA importance plot to {os.path.join(results_dir, 'aa_importance.png')}")
    
    # Print top 5
    print("\nTop 5 Important Amino Acids:")
    print(df.head(5))

if __name__ == "__main__":
    main()
