
import os
import requests
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_signal_peptide_uniprot(uniprot_id):
    """
    Calls the UniProt API to get the signal peptide range.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            for feature in features:
                if feature.get("type") == "Signal":
                    location = feature.get("location", {})
                    start = location.get("start", {}).get("value")
                    end = location.get("end", {}).get("value")
                    if start and end:
                        return (int(start), int(end))
        elif response.status_code == 404:
            return None # Not found
    except Exception as e:
        # print(f"Error fetching {uniprot_id}: {e}")
        pass
    return None

def main():
    raw_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/抗血栓活性肽fasta"
    out_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/mature_peptides_le_100"
    log_file = os.path.join(out_dir, "processing_log.txt")
    cache_file = os.path.join(out_dir, "signal_peptide_cache.json")
    
    unique_data = {}
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(".fasta"):
                path = os.path.join(root, file)
                sequences = parse_fasta(path)
                for header, seq in sequences:
                    if seq not in unique_data:
                        unique_data[seq] = header

    print(f"Total unique sequences found: {len(unique_data)}")
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            signal_peptide_cache = json.load(f)
    else:
        signal_peptide_cache = {}

    ids_to_fetch = []
    for seq, header in unique_data.items():
        if "|" in header:
            parts = header.split("|")
            if len(parts) >= 2:
                uniprot_id = parts[1]
                if uniprot_id not in signal_peptide_cache:
                    ids_to_fetch.append(uniprot_id)

    print(f"Need to fetch {len(ids_to_fetch)} IDs from UniProt API...")
    
    # Use ThreadPoolExecutor for parallel API calls
    max_workers = 20 # Adjust as needed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(get_signal_peptide_uniprot, uid): uid for uid in ids_to_fetch}
        for future in tqdm(as_completed(future_to_id), total=len(ids_to_fetch), desc="Fetching from UniProt"):
            uid = future_to_id[future]
            try:
                signal_info = future.result()
                signal_end = signal_info[1] if signal_info else 0
                signal_peptide_cache[uid] = signal_end
            except Exception as e:
                # print(f"Error processing {uid}: {e}")
                signal_peptide_cache[uid] = 0
            
            if len(signal_peptide_cache) % 100 == 0:
                with open(cache_file, "w") as f:
                    json.dump(signal_peptide_cache, f)

    # Final save of cache
    with open(cache_file, "w") as f:
        json.dump(signal_peptide_cache, f)

    mature_le_100 = []
    with open(log_file, "w") as log:
        for seq, header in unique_data.items():
            uniprot_id = None
            if "|" in header:
                parts = header.split("|")
                if len(parts) >= 2:
                    uniprot_id = parts[1]
            
            signal_end = 0
            if uniprot_id:
                signal_end = signal_peptide_cache.get(uniprot_id, 0)
            
            mature_seq = seq[signal_end:]
            if len(mature_seq) <= 100:
                mature_le_100.append((header, mature_seq, signal_end))
                log.write(f"ID: {uniprot_id or 'Unknown'} | Signal End: {signal_end} | Mature Len: {len(mature_seq)} | Original Header: {header}\n")

    out_file = os.path.join(out_dir, "mature_combined_le_100.fasta")
    with open(out_file, "w") as f:
        for header, seq, _ in mature_le_100:
            f.write(f"{header}\n{seq}\n")
            
    print(f"\nProcessing complete.")
    print(f"Total sequences with mature length <= 100: {len(mature_le_100)}")
    print(f"Mature sequences saved to {out_file}")

if __name__ == "__main__":
    main()
