# Antithrombotic Peptide Predictor (ATP-Predictor)

ATP-Predictor is a deep learning-based tool designed to accurately identify antithrombotic peptides from amino acid sequences. It leverages the power of ESM-2 (Evolutionary Scale Modeling) protein language models combined with a CNN-BiLSTM-Attention hybrid architecture.

## Key Features
- **High Accuracy**: Achieving an AUC of 0.9813 through 5-fold cross-validation.
- **Deep Semantic Representation**: Utilizes ESM-2 (`esm2_t6_8M_UR50D`) for rich sequence embedding.
- **Interpretability**: Built-in Attention mechanism to visualize key functional residues (e.g., KGD/RGD motifs).
- **User-Friendly Web Interface**: Easy-to-use Gradio-based GUI for single sequence or batch prediction.
- **Scalable**: Capable of processing tens of thousands of sequences efficiently.

## Project Structure
```
├── code/                  # Core Python scripts
│   ├── app.py             # Gradio Web UI entry point
│   ├── models.py          # CNN-BiLSTM-Attention model definition
│   ├── train.py           # Model training script
│   ├── predict.py         # Batch prediction script
│   ├── extract_esm2_features.py  # ESM-2 feature extraction
│   ├── interpret_model.py        # Model interpretability analysis
│   └── analyze_*.py      # Data analysis scripts
├── checkpoints/           # Pre-trained model weights
├── results/               # Performance metrics and visualizations
├── docs/                  # Graduation thesis and running guides
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ckf111/Antithrombotic-Peptide-Predictor.git
cd Antithrombotic-Peptide-Predictor
```

### 2. Create a virtual environment (recommended)
```bash
# Option A: conda
conda create -n anti_peptide python=3.9
conda activate anti_peptide

# Option B: venv
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web UI
```bash
python code/app.py
```
Then open **http://localhost:7860** in your browser.

> On first run, the ESM-2 model (`facebook/esm2_t6_8M_UR50D`, ~30MB) will be automatically downloaded from HuggingFace. Once cached, subsequent runs will load offline.

## Troubleshooting

### HuggingFace download fails (timeout / connection error)
If you cannot access `huggingface.co` directly, set a mirror before running:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python code/app.py
```
Or if you've already downloaded the model once, the app will automatically detect the cache and run in offline mode.

### Gradio `TypeError: argument of type 'bool' is not iterable`
This is a known bug in `gradio-client 1.3.0` with `gradio 4.44.x`. Fix it by patching `gradio_client/utils.py`:

Add a type guard at the beginning of `_json_schema_to_python_type`:
```python
def _json_schema_to_python_type(schema: Any, defs) -> str:
    if not isinstance(schema, dict):
        return str(schema)
    # ... rest of the function
```
And at the beginning of `get_type`:
```python
def get_type(schema: dict):
    if not isinstance(schema, dict):
        return str(type(schema).__name__)
    # ... rest of the function
```

Alternatively, upgrade Gradio to a version that includes the fix:
```bash
pip install --upgrade gradio
```

### Checkpoint path error
The app uses relative paths from `code/` to locate checkpoints. If you move files around, make sure `checkpoints/best_model_fold1.pth` exists relative to `code/app.py`.

## License
MIT License
