
import os

def parse_fasta(file_path):
    sequences = []
    current_header = ""
    current_seq = ""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append((current_header, current_seq))
                    current_seq = ""
                current_header = line
            else:
                current_seq += line
        if current_seq:
            sequences.append((current_header, current_seq))
    return sequences

def main():
    raw_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/抗血栓活性肽fasta"
    out_base = "/home/SCS2026004/ckf_workspace2/code/raw_data"
    
    thresholds = [50, 100, 150]
    # Use dict to store unique sequences for each threshold
    unique_data = {t: {} for t in thresholds}
    
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(".fasta"):
                path = os.path.join(root, file)
                sequences = parse_fasta(path)
                for header, seq in sequences:
                    for t in thresholds:
                        if len(seq) <= t:
                            # Keep first header encountered for each unique sequence
                            if seq not in unique_data[t]:
                                unique_data[t][seq] = header

    for t in thresholds:
        out_dir = os.path.join(out_base, f"len_le_{t}")
        out_file = os.path.join(out_dir, f"combined_le_{t}.fasta")
        with open(out_file, "w") as f:
            for seq, header in unique_data[t].items():
                f.write(f"{header}\n{seq}\n")
        print(f"Threshold {t}: Found {len(unique_data[t])} unique sequences. Saved to {out_file}")

if __name__ == "__main__":
    main()
