# Antithrombotic Peptide Predictor (ATP-Predictor)

ATP-Predictor is a deep learning-based tool designed to accurately identify antithrombotic peptides from amino acid sequences. It leverages the power of ESM-2 (Evolutionary Scale Modeling) protein language models combined with a CNN-BiLSTM-Attention hybrid architecture.

## Key Features
- **High Accuracy**: Achieving an AUC of 0.9813 through 5-fold cross-validation.
- **Deep Semantic Representation**: Utilizes ESM-2 (`esm2_t6_8M_UR50D`) for rich sequence embedding.
- **Interpretability**: Built-in Attention mechanism to visualize key functional residues (e.g., KGD/RGD motifs).
- **User-Friendly Web Interface**: Easy-to-use Gradio-based GUI for single sequence or batch prediction.
- **Scalable**: Capable of processing tens of thousands of sequences efficiently.

## Project Structure
- `code/`: Core Python scripts for training, inference, and feature extraction.
- `checkpoints/`: Pre-trained model weights.
- `results/`: Performance metrics and interpretability visualizations.
- `docs/`: Full graduation thesis and detailed running guides.

## Quick Start
1. **Environment Setup**:
   ```bash
   conda create -n anti_peptide python=3.9
   conda activate anti_peptide
   pip install torch transformers gradio pandas numpy matplotlib seaborn
   ```
2. **Run Web UI**:
   ```bash
   python code/app.py
   ```

## License
MIT License
