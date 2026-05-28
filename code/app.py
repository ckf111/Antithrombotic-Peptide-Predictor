import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置环境变量
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 检查 ESM-2 模型是否已缓存，已缓存则启用离线模式避免网络请求
_hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
_esm_model_name = "facebook/esm2_t6_8M_UR50D"
if os.path.isdir(os.path.join(_hf_cache_dir, "models--" + _esm_model_name.replace("/", "--"))):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, EsmModel
from models import HybridModel
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
from PIL import Image

# 配置学术绘图
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

class PredictorApp:
    def __init__(self, model_path, esm_model_name="facebook/esm2_t6_8M_UR50D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.esm_model_name = esm_model_name
        
        # 懒加载标志
        self.tokenizer = None
        self.esm_model = None
        self.model = None
        
        print(f"App initialized (Lazy Loading enabled). Using device: {self.device}")

    def load_resources(self):
        """懒加载模型资源以节省初始内存"""
        if self.tokenizer is None:
            print(f"Loading ESM-2 Tokenizer ({self.esm_model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
        
        if self.esm_model is None:
            print(f"Loading ESM-2 Model ({self.esm_model_name})...")
            self.esm_model = EsmModel.from_pretrained(self.esm_model_name).to(self.device)
            self.esm_model.eval()
            
        if self.model is None:
            print(f"Loading Hybrid Model from {self.model_path}...")
            self.model = HybridModel(input_dim=320, cnn_channels=64, lstm_hidden=128, dropout=0.3).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

    def plot_to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        return img

    def visualize_attention(self, seq, weights, prob):
        seq_len = len(seq)
        weights = weights[:seq_len]
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, seq_len * 0.3), 6), 
                                       gridspec_kw={'height_ratios': [1, 2]})
        
        # Heatmap
        sns.heatmap([norm_weights], annot=[list(seq)], fmt="", cmap="magma", cbar=False,
                    yticklabels=False, xticklabels=False, ax=ax1)
        ax1.set_title(f"Attention Heatmap (Prob: {prob:.4f})", fontsize=12)
        
        # Line plot
        x_pos = np.arange(1, seq_len + 1)
        ax2.plot(x_pos, weights, marker='o', markersize=4, color='#d62728', linewidth=1.5)
        ax2.fill_between(x_pos, weights, alpha=0.15, color='#d62728')
        
        # Highlight motifs
        for motif in ["KGD", "RGD", "FGD"]:
            start = seq.find(motif)
            while start != -1:
                ax2.axvspan(start + 1, start + len(motif), color='yellow', alpha=0.3)
                start = seq.find(motif, start + 1)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(list(seq), fontsize=8)
        ax2.set_ylabel("Attention Score")
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        img = self.plot_to_image(fig)
        plt.close(fig)
        return img

    def predict(self, sequence):
        sequence = sequence.strip().upper()
        if not sequence or any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence):
            return "❌ 错误: 无效的氨基酸序列。仅允许标准 20 种氨基酸。", None, None
        
        if len(sequence) > 100:
            return "❌ 错误: 序列过长 (演示版最大限制 100)。", None, None

        # 触发懒加载
        self.load_resources()

        start_time = time.time()
        with torch.no_grad():
            # 1. ESM-2 Features
            inputs = self.tokenizer(sequence, return_tensors="pt", padding="max_length", 
                                  max_length=102, truncation=True).to(self.device)
            esm_out = self.esm_model(**inputs)
            embeddings = esm_out.last_hidden_state[:, 1:len(sequence)+1, :]
            
            # Pad to 100 for the hybrid model
            padded_feat = torch.zeros((1, 100, 320)).to(self.device)
            padded_feat[0, :len(sequence), :] = embeddings[0]
            
            # 2. Prediction
            prob, weights = self.model(padded_feat)
            prob_val = prob.item()
            weights_val = weights.squeeze().cpu().numpy()
            
        elapsed = time.time() - start_time
        
        # Result text
        label = "✅ 抗血栓活性肽 (高活性可能性)" if prob_val >= 0.5 else "❌ 非抗血栓肽 (低活性可能性)"
        result_text = f"### 预测结果: {label}\n\n**置信度得分:** `{prob_val:.4f}`\n\n**推理耗时:** `{elapsed:.4f}s`"
        
        # Visualization
        viz_img = self.visualize_attention(sequence, weights_val, prob_val)
        
        # Top residues
        top_indices = np.argsort(weights_val[:len(sequence)])[-3:][::-1]
        top_res_text = ", ".join([f"{sequence[idx]}{idx+1}" for idx in top_indices])
        
        return result_text, viz_img, f"关键残基定位: {top_res_text}"

# Initialize App
checkpoint = os.path.join(BASE_DIR, "../checkpoints/best_model_fold1.pth")
app = PredictorApp(checkpoint)

# Build Interface
with gr.Blocks(title="Antithrombotic Peptide Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧬 基于深度学习的抗血栓肽智能预测系统
    本系统整合了 **ESM-2** 预训练语言模型与 **CNN-BiLSTM-Attention** 混合架构，提供端到端的多肽活性预测与位点分析。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_seq = gr.Textbox(
                label="输入多肽序列 (Amino Acid Sequence)", 
                placeholder="例如: VVYTDCTESGQNLCLCEGSNVCGQGNKCILGSDGEKNQCVTGEGTPKPQSHNDGDFEEIPEEYLQ",
                lines=3
            )
            btn = gr.Button("开始预测 (Predict)", variant="primary")
            
            gr.Examples(
                examples=[
                    ["VVYTDCTESGQNLCLCEGSNVCGQGNKCILGSDGEKNQCVTGEGTPKPQSHNDGDFEEIPEEYLQ"], # Hirudin
                    ["EECONPCKTCKCPREKCNCPGEKCKCDRDGQKC"], # Eristostatin snippet
                    ["QCVTGEGTPKPQSHNDGDFEEIPEEYLQ"] # C-term Hirudin
                ],
                inputs=input_seq
            )
            
        with gr.Column(scale=1):
            out_text = gr.Markdown(label="预测详情")
            out_top = gr.Label(label="关键位点分析")

    with gr.Row():
        out_plot = gr.Image(label="注意力机制权重图谱 (Attention Visualization)")

    btn.click(fn=app.predict, inputs=input_seq, outputs=[out_text, out_plot, out_top])

if __name__ == "__main__":
    # 使用 share=True 自动处理端口冲突并提供外网访问地址
    demo.launch(server_name="0.0.0.0", share=True)
