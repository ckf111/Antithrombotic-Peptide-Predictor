
import os

def parse_fasta(file_path):
    sequences = []
    current_seq = ""
    with open(file_path, "r") as f:
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

def analyze_fasta_directory(directory):
    fasta_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".fasta"):
                fasta_files.append(os.path.join(root, file))
    
    unique_sequences = set()
    for fasta_file in fasta_files:
        sequences = parse_fasta(fasta_file)
        for seq in sequences:
            unique_sequences.add(seq)

    print("Unique Length Distribution:")
    bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 1000]
    counts = [0] * (len(bins) - 1)
    for seq in unique_sequences:
        length = len(seq)
        for i in range(len(bins) - 1):
            if bins[i] < length <= bins[i+1]:
                counts[i] += 1
                break
    
    accumulated = 0
    for i in range(len(bins) - 1):
        accumulated += counts[i]
        print(f"{bins[i]:>4} - {bins[i+1]:>4}: {counts[i]:>6} (Accumulated: {accumulated:>6})")

if __name__ == "__main__":
    data_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/抗血栓活性肽fasta"
    analyze_fasta_directory(data_dir)
