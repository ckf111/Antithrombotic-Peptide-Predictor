
import numpy as np
import os

# Standard 20 amino acids
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Physicochemical Properties (PCP) - normalized values (approximate)
# 1. Hydrophobicity (Kyte-Doolittle)
# 2. Hydrophilicity (Hopp-Woods)
# 3. Molecular Weight (normalized)
# 4. pI (Isoelectric point, normalized)
PCP_DICT = {
    'A': [1.8, -0.5, 89.1, 6.0],
    'C': [2.5, -1.0, 121.2, 5.0],
    'D': [-3.5, 3.0, 133.1, 2.8],
    'E': [-3.5, 3.0, 147.1, 3.2],
    'F': [2.8, -2.5, 165.2, 5.5],
    'G': [-0.4, 0.0, 75.1, 6.0],
    'H': [-3.2, -0.5, 155.2, 7.6],
    'I': [4.5, -1.8, 131.2, 6.0],
    'K': [-3.9, 3.0, 146.2, 9.7],
    'L': [3.8, -1.8, 131.2, 6.0],
    'M': [1.9, -1.3, 149.2, 5.7],
    'N': [-3.5, 0.2, 132.1, 5.4],
    'P': [-1.6, 0.0, 115.1, 6.3],
    'Q': [-3.5, 0.2, 146.1, 5.7],
    'R': [-4.5, 3.0, 174.2, 10.8],
    'S': [-0.8, 0.3, 105.1, 5.7],
    'T': [-0.7, -0.4, 119.1, 5.6],
    'V': [4.2, -1.5, 117.1, 6.0],
    'W': [-0.9, -3.4, 204.2, 5.9],
    'Y': [-1.3, -2.3, 181.2, 5.7]
}

# Normalize PCP values
pcp_array = np.array(list(PCP_DICT.values()))
pcp_mean = pcp_array.mean(axis=0)
pcp_std = pcp_array.std(axis=0)
NORM_PCP_DICT = {aa: (np.array(val) - pcp_mean) / pcp_std for aa, val in PCP_DICT.items()}

def one_hot_encode(sequence, max_len=100):
    encoding = np.zeros((max_len, 20))
    for i, aa in enumerate(sequence[:max_len]):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1
    return encoding

def pcp_encode(sequence, max_len=100):
    encoding = np.zeros((max_len, 4))
    for i, aa in enumerate(sequence[:max_len]):
        if aa in NORM_PCP_DICT:
            encoding[i] = NORM_PCP_DICT[aa]
    return encoding

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

def process_and_save(fasta_path, label, prefix):
    seqs = parse_fasta(fasta_path)
    one_hot_list = []
    pcp_list = []
    labels = []
    
    for seq in seqs:
        one_hot_list.append(one_hot_encode(seq))
        pcp_list.append(pcp_encode(seq))
        labels.append(label)
        
    out_dir = "/home/SCS2026004/ckf_workspace2/code/features"
    np.save(os.path.join(out_dir, f"{prefix}_one_hot.npy"), np.array(one_hot_list))
    np.save(os.path.join(out_dir, f"{prefix}_pcp.npy"), np.array(pcp_list))
    np.save(os.path.join(out_dir, f"{prefix}_labels.npy"), np.array(labels))
    print(f"Processed {len(seqs)} sequences for {prefix}. Saved to {out_dir}")

def main():
    pos_fasta = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/positive_samples.fasta"
    neg_fasta = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/negative_samples.fasta"
    
    process_and_save(pos_fasta, 1, "pos")
    process_and_save(neg_fasta, 0, "neg")

if __name__ == "__main__":
    main()
