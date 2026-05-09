
import os

def parse_fasta(file_path):
    sequences = {}
    current_header = ""
    current_seq = ""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_seq:
                    sequences[current_seq] = current_header
                    current_seq = ""
                current_header = line
            else:
                current_seq += line
        if current_seq:
            sequences[current_seq] = current_header
    return sequences

def main():
    raw_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/抗血栓活性肽fasta"
    v2_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/mature_peptides_le_100-v2"
    os.makedirs(v2_dir, exist_ok=True)
    
    all_unique = {}
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(".fasta"):
                path = os.path.join(root, file)
                seqs = parse_fasta(path)
                for s, h in seqs.items():
                    if s not in all_unique:
                        all_unique[s] = h
    
    input_file = os.path.join(v2_dir, "all_unique_for_prediction.fasta")
    with open(input_file, "w") as f:
        for i, (seq, header) in enumerate(all_unique.items()):
            # Use a simplified ID for deepsig processing to avoid issues with long headers
            f.write(f">seq_{i}\n{seq}\n")
            
    # Save a mapping for later reconstruction
    mapping_file = os.path.join(v2_dir, "id_mapping.txt")
    with open(mapping_file, "w") as f:
        for i, (seq, header) in enumerate(all_unique.items()):
            f.write(f"seq_{i}\t{header}\t{seq}\n")
            
    print(f"Prepared {len(all_unique)} sequences for prediction in {input_file}")

if __name__ == "__main__":
    main()
