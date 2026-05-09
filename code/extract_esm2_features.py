
import os
# 设置镜像源（关键！）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # 增加超时时间

import torch
from transformers import AutoTokenizer, EsmModel
import numpy as np
import os

def parse_fasta(file_path):
    sequences = []
    with open(file_path, "r") as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)
    return sequences

def extract_esm2_features(sequences, model_name="facebook/esm2_t6_8M_UR50D", max_len=100):
    print(f"Loading model {model_name}...")
    print(f"Using endpoint: {os.environ.get('HF_ENDPOINT', 'default')}")
    
    try:
        # 允许断点续传
        tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True)
        print("✓ Tokenizer loaded")
        
        model = EsmModel.from_pretrained(model_name, resume_download=True)
        print("✓ Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try: export HF_ENDPOINT=https://hf-mirror.com ")
        raise
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    features = []
    total = len(sequences)
    
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if (i+1) % 100 == 0 or i == 0:
                print(f"Processing {i+1}/{total}...")
            
            # 序列长度裁剪
            seq = seq[:max_len]
            # ESM-2 tokenizer: [CLS] seq [EOS]
            inputs = tokenizer(seq, return_tensors="pt", padding="max_length", 
                           max_length=max_len+2, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # 提取残基级别的嵌入，排除 CLS (0) 和 EOS (L+1)
            embeddings = outputs.last_hidden_state[0, 1:max_len+1, :].cpu().numpy()
            features.append(embeddings)
            
    return np.array(features)

def main():
    pos_fasta = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/positive_samples.fasta"
    neg_fasta = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/negative_samples.fasta"
    out_dir = "/home/SCS2026004/ckf_workspace2/code/features"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading positive samples from {pos_fasta}...")
    pos_seqs = parse_fasta(pos_fasta)
    print(f"Found {len(pos_seqs)} positive sequences")
    
    print(f"Loading negative samples from {neg_fasta}...")
    neg_seqs = parse_fasta(neg_fasta)
    print(f"Found {len(neg_seqs)} negative sequences")
    
    print(f"\nStarting positive samples ({len(pos_seqs)})...")
    pos_esm2 = extract_esm2_features(pos_seqs)
    np.save(os.path.join(out_dir, "pos_esm2.npy"), pos_esm2)
    print(f"Saved: {os.path.join(out_dir, 'pos_esm2.npy')}")
    
    print(f"\nStarting negative samples ({len(neg_seqs)})...")
    neg_esm2 = extract_esm2_features(neg_seqs)
    np.save(os.path.join(out_dir, "neg_esm2.npy"), neg_esm2)
    print(f"Saved: {os.path.join(out_dir, 'neg_esm2.npy')}")
    
    print(f"\n✓ All ESM-2 features extracted and saved to {out_dir}")

if __name__ == "__main__":
    main()
