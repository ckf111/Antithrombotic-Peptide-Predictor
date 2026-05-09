
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
                    sequences[current_header[1:]] = current_seq
                    current_seq = ""
                current_header = line
            else:
                current_seq += line
        if current_seq:
            sequences[current_header[1:]] = current_seq
    return sequences

def main():
    v2_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/mature_peptides_le_100-v2"
    mapping_file = os.path.join(v2_dir, "id_mapping.txt")
    signalp_output_fasta = os.path.join(v2_dir, "signalp_output/processed_entries.fasta")
    prediction_results = os.path.join(v2_dir, "signalp_output/prediction_results.txt")
    
    # Load mapping: seq_id -> (original_header, original_seq)
    id_map = {}
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                id_map[parts[0]] = (parts[1], parts[2])
                
    # Load SignalP mature sequences
    # Note: SignalP 6.0 processed_entries.fasta contains mature sequences for those with SP,
    # and original sequences for those predicted as OTHER (No SP).
    # Wait, let's verify this. If OTHER, is it included in processed_entries.fasta?
    # README says: "Predicted mature proteins, i.e. sequences with their signal peptides removed."
    # Let's check seq_2 (OTHER in prediction_results.txt) in processed_entries.fasta
    
    mature_seqs = parse_fasta(signalp_output_fasta)
    
    final_mature_le_100 = []
    
    # We want to iterate through ALL sequences and apply the "mature" version.
    # For OTHER, the "mature" version is just the original sequence.
    
    for seq_id, (header, original_seq) in id_map.items():
        if seq_id in mature_seqs:
            m_seq = mature_seqs[seq_id]
        else:
            # If not in mature_seqs, it might be because SignalP didn't output it?
            # Usually SignalP 6.0 outputs all sequences in processed_entries.fasta.
            m_seq = original_seq
            
        if len(m_seq) <= 100:
            final_mature_le_100.append((header, m_seq))
            
    # Save the results
    out_file = os.path.join(v2_dir, "mature_combined_le_100_v2.fasta")
    with open(out_file, "w") as f:
        for header, seq in final_mature_le_100:
            f.write(f"{header}\n{seq}\n")
            
    print(f"Total unique sequences in V2 (SignalP 6): {len(final_mature_le_100)}")
    
    # Compare with V1
    v1_file = "/home/SCS2026004/ckf_workspace2/code/raw_data/mature_peptides_le_100-v1/mature_combined_le_100.fasta"
    if os.path.exists(v1_file):
        v1_count = 0
        with open(v1_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    v1_count += 1
        print(f"Total unique sequences in V1 (UniProt): {v1_count}")
        print(f"Difference: {len(final_mature_le_100) - v1_count}")

if __name__ == "__main__":
    main()
