import pandas as pd

# Define file paths
input_file = "/home/SCS2026004/ckf_workspace2/code/raw_data/new-test-data/new_peptides_predictions.csv"
output_file = "/home/SCS2026004/ckf_workspace2/code/raw_data/new-test-data/relabelled_peptides_final.csv"

def relabel_sequences():
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Create the result dataframe
    # 1. Sequential index starting from 1
    # 2. Sequence
    # 3. Label based on Probability > 0.95
    
    result_df = pd.DataFrame()
    result_df['序号'] = range(1, len(df) + 1)
    result_df['多肽序列'] = df['Sequence']
    
    # Apply labeling logic: > 0.95 is Positive, else Negative
    result_df['标签'] = df['Probability'].apply(lambda x: "正样本" if x > 0.95 else "负样本")
    
    # Save to CSV
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Print summary
    pos_count = (result_df['标签'] == "正样本").sum()
    neg_count = (result_df['标签'] == "负样本").sum()
    print(f"Processing complete.")
    print(f"Total sequences: {len(result_df)}")
    print(f"Positive samples (>0.95): {pos_count}")
    print(f"Negative samples (<=0.95): {neg_count}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    relabel_sequences()
