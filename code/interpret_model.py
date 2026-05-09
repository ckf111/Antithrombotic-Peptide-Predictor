import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models import HybridModel
import pandas as pd

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

def parse_fasta_with_headers(file_path):
    entries = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        current_header = ""
        current_seq = ""
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_header:
                    entries.append((current_header, current_seq))
                current_header = line[1:]
                current_seq = ""
            else:
                current_seq += line
        if current_header:
            entries.append((current_header, current_seq))
    return entries

def visualize_attention(seq, weights, prob, title, save_path):
    """
    生成精美的学术热图与折线图组合
    """
    seq_len = len(seq)
    weights = weights[:seq_len]
    
    # 归一化权重到 0-1 范围，方便热图展示
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, seq_len * 0.4), 7), 
                                   gridspec_kw={'height_ratios': [1, 2]})
    
    # 1. 精美热图 (Heatmap)
    # 使用 'magma' 或 'YlOrRd'，更符合学术审美
    sns.heatmap([norm_weights], annot=[list(seq)], fmt="", cmap="magma", cbar=True,
                yticklabels=False, xticklabels=False, ax=ax1,
                cbar_kws={"orientation": "horizontal", "pad": 0.15, "label": "Normalized Attention"})
    ax1.set_title(f"Sequence: {title} (Prediction Prob: {prob:.4f})", fontsize=14, pad=15)
    
    # 2. 折线图 (Line plot)
    x_pos = np.arange(1, seq_len + 1)
    ax2.plot(x_pos, weights, marker='o', markersize=6, linestyle='-', color='#d62728', linewidth=2, label='Attention Weight')
    ax2.fill_between(x_pos, weights, alpha=0.2, color='#d62728')
    
    # 标注高权重残基 (Top 3)
    top_indices = np.argsort(weights)[-3:]
    for idx in top_indices:
        ax2.annotate(f"{seq[idx]}{idx+1}", (idx+1, weights[idx]), 
                     textcoords="offset points", xytext=(0,10), ha='center',
                     fontsize=10, fontweight='bold', color='darkblue')

    # 检查是否有已知基序 (如 KGD, RGD) 并突出显示
    known_motifs = ["KGD", "RGD", "FGD"]
    for motif in known_motifs:
        start_idx = seq.find(motif)
        while start_idx != -1:
            ax2.axvspan(start_idx + 1, start_idx + len(motif), color='yellow', alpha=0.3, label=f'Motif: {motif}')
            start_idx = seq.find(motif, start_idx + 1)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(seq), fontsize=10)
    ax2.set_xlabel("Amino Acid Residue Position", fontsize=12)
    ax2.set_ylabel("Attention Score", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(0, max(weights) * 1.2)
    
    plt.tight_layout()
    # 同时保存 PNG 和 PDF (PDF 适合插入论文，无损)
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualizations to {save_path}.png/pdf")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载模型
    model = HybridModel(input_dim=320, cnn_channels=64, lstm_hidden=128, dropout=0.3).to(device)
    # 使用最佳模型路径
    checkpoint_path = "/home/SCS2026004/ckf_workspace2/code/checkpoints/best_model_fold1.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. 加载数据
    pos_fasta = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/positive_samples.fasta"
    pos_entries = parse_fasta_with_headers(pos_fasta)
    
    # 检查特征文件是否存在
    feat_path = "/home/SCS2026004/ckf_workspace2/code/features/pos_esm2.npy"
    if not os.path.exists(feat_path):
        print(f"Error: Features {feat_path} not found. Run feature extraction first.")
        return
    pos_esm2 = np.load(feat_path)
    
    # 3. 选择具有代表性的多肽进行分析
    # 这些是抗血栓肽研究中经典的分子
    targets = {
        "P01050": "Hirudin",       # 水蛭素 (最强效的凝血酶抑制剂)
        "P0C6S4": "Eristostatin",  # 具有 KGD 基序的去整合素
        "P28375": "Dendroaspin",   # 蛇毒中的去整合素
        "P02714": "Antistasin",    # 抗凝血肽
    }
    
    output_dir = "/home/SCS2026004/ckf_workspace2/code/results/interpretability"
    os.makedirs(output_dir, exist_ok=True)
    
    interpret_data = []

    with torch.no_grad():
        for i, (header, seq) in enumerate(pos_entries):
            found_key = None
            for key in targets.keys():
                if key in header:
                    found_key = key
                    break
            
            if found_key:
                print(f"Analyzing {targets[found_key]} ({header})...")
                
                # 获取 ESM-2 特征
                feat = torch.FloatTensor(pos_esm2[i]).unsqueeze(0).to(device)
                
                # 模型推理
                prob, weights = model(feat)
                prob_val = prob.item()
                weights_val = weights.squeeze().cpu().numpy()
                
                # 可视化
                name = targets[found_key]
                save_path = os.path.join(output_dir, f"interpret_{name}")
                visualize_attention(seq, weights_val, prob_val, name, save_path)
                
                # 记录关键残基
                seq_len = len(seq)
                w_trimmed = weights_val[:seq_len]
                top_indices = np.argsort(w_trimmed)[-5:][::-1]
                top_residues = [f"{seq[idx]}{idx+1}" for idx in top_indices]
                
                interpret_data.append({
                    "Name": name,
                    "Accession": found_key,
                    "Probability": prob_val,
                    "Top_Residues": ", ".join(top_residues),
                    "Sequence": seq
                })

    # 保存分析汇总表
    if interpret_data:
        df = pd.DataFrame(interpret_data)
        df.to_csv(os.path.join(output_dir, "interpretation_summary.csv"), index=False)
        print(f"\n✓ Interpretation summary saved to {output_dir}/interpretation_summary.csv")

if __name__ == "__main__":
    main()
