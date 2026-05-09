
import os
import requests
import random
import time

def parse_fasta(file_path):
    sequences = set()
    with open(file_path, "r") as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.add(current_seq)
                    current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.add(current_seq)
    return sequences

def fetch_negative_samples(num_needed=1000):
    # Simpler query to avoid 400 errors
    # Search for reviewed sequences between 10 and 100 AA
    # We will filter keywords and duplicates locally to be safe
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": "reviewed:true AND length:[10 TO 100]",
        "format": "fasta",
        "size": 500
    }
    
    all_negatives = []
    seen_seqs = set()
    
    print(f"Fetching sequences from UniProt...")
    
    # Pagination loop
    while len(all_negatives) < num_needed:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            content = response.text
            lines = content.split('\n')
            current_header = ""
            current_seq = ""
            for line in lines:
                if line.startswith(">"):
                    if current_header and current_seq:
                        if current_seq not in seen_seqs:
                            all_negatives.append((current_header, current_seq))
                            seen_seqs.add(current_seq)
                    current_header = line
                    current_seq = ""
                else:
                    current_seq += line
            if current_header and current_seq:
                if current_seq not in seen_seqs:
                    all_negatives.append((current_header, current_seq))
                    seen_seqs.add(current_seq)
            
            print(f"Current count: {len(all_negatives)}")
            
            # Check for Link header for next page
            if 'Link' in response.headers:
                links = response.headers['Link'].split(',')
                next_url = None
                for link in links:
                    if 'rel="next"' in link:
                        next_url = link.split(';')[0].strip('<>')
                        break
                if next_url:
                    url = next_url
                    params = {} # params are already in the next_url
                else:
                    break
            else:
                break
        else:
            print(f"Error fetching data: {response.status_code}")
            break
            
    return all_negatives

def main():
    pos_file = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/positive_samples.fasta"
    neg_file = "/home/SCS2026004/ckf_workspace2/code/final_raw_data/negative_samples.fasta"
    
    pos_seqs = parse_fasta(pos_file)
    print(f"Loaded {len(pos_seqs)} positive sequences.")

    raw_negatives = fetch_negative_samples(num_needed=1500)
    
    # Filter out sequences that might have antithrombotic activity based on header keywords
    keywords_to_exclude = ["anticoagulant", "antithrombotic", "thrombin", "platelet", "coagulation", "fibrin"]
    
    final_negatives = []
    for header, seq in raw_negatives:
        header_lower = header.lower()
        if any(kw in header_lower for kw in keywords_to_exclude):
            continue
        if seq in pos_seqs:
            continue
        final_negatives.append((header, seq))
        if len(final_negatives) >= 1000:
            break
            
    print(f"Final negative samples count: {len(final_negatives)}")
    
    with open(neg_file, "w") as f:
        for header, seq in final_negatives:
            f.write(f"{header}\n{seq}\n")
            
    print(f"Saved to {neg_file}")

if __name__ == "__main__":
    main()
