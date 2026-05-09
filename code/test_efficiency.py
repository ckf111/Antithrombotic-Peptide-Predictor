
import os
# Set mirror at the very beginning
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1" # Try to use local files only if possible

import time
import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from models import HybridModel

def test_efficiency():
    model_name = "facebook/esm2_t6_8M_UR50D"
    max_len = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing efficiency on device: {device}")
    
    # 1. Load ESM-2
    print("Loading ESM-2 (Offline Mode)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        esm_model = EsmModel.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception as e:
        print(f"Offline load failed, trying online with mirror: {e}")
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        esm_model = EsmModel.from_pretrained(model_name, local_files_only=False).to(device)
        
    esm_model.eval()
    
    # 2. Load Hybrid Model
    print("Loading Hybrid Model...")
    hybrid_model = HybridModel(input_dim=320, cnn_channels=64, lstm_hidden=128, dropout=0.3).to(device)
    checkpoint_path = "/home/SCS2026004/ckf_workspace2/code/checkpoints/best_model_fold1.pth"
    if os.path.exists(checkpoint_path):
        hybrid_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    hybrid_model.eval()
    
    # 3. Test sequence
    test_seq = "MVLSAADKGNVKAAWGKVGGHAAEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR" 
    test_seq = test_seq[:max_len]
    
    print(f"\nTesting prediction for 1 sequence: {test_seq}")
    
    # Warm up
    with torch.no_grad():
        inputs = tokenizer(test_seq, return_tensors="pt", padding="max_length", max_length=max_len+2, truncation=True).to(device)
        embeddings = esm_model(**inputs).last_hidden_state[:, 1:max_len+1, :]
        _ = hybrid_model(embeddings)[0]
    
    # Measure time
    n_runs = 50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            inputs = tokenizer(test_seq, return_tensors="pt", padding="max_length", max_length=max_len+2, truncation=True).to(device)
            embeddings = esm_model(**inputs).last_hidden_state[:, 1:max_len+1, :]
            probs, _ = hybrid_model(embeddings)
            _ = probs.cpu().numpy()
            
    avg_time = (time.time() - start_time) / n_runs
    print(f"\nAverage prediction time per sequence: {avg_time:.4f} seconds")
    
    if avg_time < 2.0:
        print(f"✓ Efficiency requirement (< 2s) met! (Actual: {avg_time:.4f}s)")
    else:
        print(f"✗ Efficiency requirement (< 2s) NOT met. (Actual: {avg_time:.4f}s)")

if __name__ == "__main__":
    test_efficiency()
