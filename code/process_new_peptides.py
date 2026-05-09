import pandas as pd
import os
import subprocess

# 1. Load the new peptides CSV
input_csv = "/home/SCS2026004/ckf_workspace2/code/raw_data/new-test-data/combined_200_peptides.csv"
output_dir = "/home/SCS2026004/ckf_workspace2/code/raw_data/new-test-data"
temp_fasta = os.path.join(output_dir, "temp_for_prediction.fasta")

df = pd.read_csv(input_csv)

# 2. Convert to FASTA format for predict.py
with open(temp_fasta, "w") as f:
    for idx, row in df.iterrows():
        # Use protein_id and peptide info as header
        header = f">{row['protein_id']}_idx_{idx}"
        f.write(f"{header}\n{row['peptide']}\n")

print(f"Created temporary FASTA for {len(df)} sequences.")

# 3. Run prediction using the anti_peptide environment
# Note: We use conda run to ensure correct environment
predict_script = "/home/SCS2026004/ckf_workspace2/code/predict.py"
predictions_csv = os.path.join(output_dir, "new_peptides_predictions.csv")

cmd = [
    "conda", "run", "-n", "anti_peptide", "python", predict_script,
    "--fasta", temp_fasta,
    "--output", predictions_csv
]

print("Running predictions...")
env = os.environ.copy()
env["HF_ENDPOINT"] = "https://hf-mirror.com"
subprocess.run(cmd, env=env, check=True)

# 4. Filter antithrombotic peptides
pred_df = pd.read_csv(predictions_csv)
# Merge back original info if needed, or just filter
# The 'Header' in pred_df matches '>protein_id_idx_idx' without '>'
pred_df['original_idx'] = pred_df['Header'].apply(lambda x: int(x.split('_idx_')[-1]))

# Filter based on Prediction label
antithrombotic_peptides = pred_df[pred_df['Prediction'] == 'Antithrombotic'].copy()

# Save final filtered results
final_output = os.path.join(output_dir, "antithrombotic_peptides_filtered.csv")
antithrombotic_peptides.to_csv(final_output, index=False)

print(f"Filtered {len(antithrombotic_peptides)} antithrombotic peptides.")
print(f"Results saved to: {final_output}")

# Cleanup
if os.path.exists(temp_fasta):
    os.remove(temp_fasta)
