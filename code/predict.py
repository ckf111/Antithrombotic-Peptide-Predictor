import os
# 设置环境变量必须在导入 transformers 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from models import HybridModel
import time

def parse_fasta(file_path):
    entries = []
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

def predict_batch(sequences, model, esm_model, tokenizer, device, max_len=100, batch_size=16):
    model.eval()
    esm_model.eval()
    
    all_probs = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        
        # Truncate to max_len
        batch_seqs_trunc = [s[:max_len] for s in batch_seqs]
        
        with torch.no_grad():
            inputs = tokenizer(batch_seqs_trunc, return_tensors="pt", padding="max_length", 
                             max_length=max_len+2, truncation=True).to(device)
            
            # Extract ESM-2 features
            outputs = esm_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 1:max_len+1, :]
            
            # Hybrid model prediction
            probs, _ = model(embeddings)
            
            # If batch_size=1, probs might be a scalar, make it a list
            if len(batch_seqs) == 1:
                all_probs.append(probs.item())
            else:
                all_probs.extend(probs.cpu().numpy().tolist())
                
    return all_probs

def main():
    parser = argparse.ArgumentParser(description="Antithrombotic Peptide Predictor")
    parser.add_argument("--seq", type=str, help="Single protein sequence to predict")
    parser.add_argument("--fasta", type=str, help="Path to FASTA file for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save results (for batch prediction)")
    parser.add_argument("--checkpoint", type=str, default="/home/SCS2026004/ckf_workspace2/code/checkpoints/best_model_fold1.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    if not args.seq and not args.fasta:
        parser.print_help()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # 2. Load Models
    model_name = "facebook/esm2_t6_8M_UR50D"
    print(f"Loading ESM-2 ({model_name})...", flush=True)
    
    try:
        # 首先尝试离线加载以加快速度
        print("Attempting to load tokenizer offline...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✓ Tokenizer loaded offline!", flush=True)
        
        print("Attempting to load ESM model offline...", flush=True)
        esm_model = EsmModel.from_pretrained(model_name, local_files_only=True).to(device)
        print("✓ ESM model loaded offline!", flush=True)
    except Exception:
        print("Offline load failed, trying online with mirror (this might take a few minutes)...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        print("✓ Tokenizer loaded online!", flush=True)
        esm_model = EsmModel.from_pretrained(model_name, local_files_only=False).to(device)
        print("✓ ESM model loaded online!", flush=True)
    
    print(f"Loading Hybrid Model from {args.checkpoint}...", flush=True)
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}", flush=True)
        return
    
    hybrid_model = HybridModel(input_dim=320, cnn_channels=64, lstm_hidden=128, dropout=0.3).to(device)
    hybrid_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("✓ Hybrid model loaded!", flush=True)
    
    # 3. Predict
    if args.seq:
        print(f"\nPredicting single sequence: {args.seq[:30]}...", flush=True)
        start_time = time.time()
        probs = predict_batch([args.seq], hybrid_model, esm_model, tokenizer, device)
        prob = probs[0]
        label = "Antithrombotic" if prob >= args.threshold else "Non-Antithrombotic"
        print(f"Prediction Result: {label}", flush=True)
        print(f"Probability Score: {prob:.4f}", flush=True)
        print(f"Time taken: {time.time() - start_time:.4f}s", flush=True)
        
    elif args.fasta:
        print(f"\nLoading FASTA from {args.fasta}...", flush=True)
        entries = parse_fasta(args.fasta)
        headers = [e[0] for e in entries]
        sequences = [e[1] for e in entries]
        print(f"Found {len(sequences)} sequences.", flush=True)
        
        start_time = time.time()
        probs = predict_batch(sequences, hybrid_model, esm_model, tokenizer, device)
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame({
            "Header": headers,
            "Sequence": sequences,
            "Probability": probs,
            "Prediction": ["Antithrombotic" if p >= args.threshold else "Non-Antithrombotic" for p in probs]
        })
        df.to_csv(args.output, index=False)
        print(f"\nBatch prediction complete!", flush=True)
        print(f"Results saved to: {args.output}", flush=True)
        print(f"Total time taken for {len(sequences)} sequences: {time.time() - start_time:.4f}s", flush=True)
        print(f"Average time per sequence: {(time.time() - start_time)/len(sequences):.4f}s", flush=True)

if __name__ == "__main__":
    print("Script started", flush=True)
    main()
