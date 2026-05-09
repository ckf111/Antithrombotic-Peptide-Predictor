# 抗血栓活性肽预测项目：代码运行指南

本项目提供了一套完整的从数据预处理、特征提取到模型训练及 Web 预测系统的解决方案。以下是详细的运行步骤。

## 1. 环境准备

推荐使用 `conda` 创建虚拟环境以避免依赖冲突。

### 1.1 创建虚拟环境
```bash
conda create -n anti_peptide python=3.9 -y
conda activate anti_peptide
```

### 1.2 安装依赖
```bash
# 安装 PyTorch (根据显卡版本选择，这里以通用版为例)
pip install torch torchvision torchaudio

# 安装其他核心依赖
pip install transformers gradio numpy pandas scikit-learn matplotlib seaborn Pillow biopython
```

### 1.3 设置国内镜像源 (针对 ESM-2 模型下载)
由于模型文件较大且托管在 Hugging Face，建议运行前设置镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 数据准备

在运行训练脚本前，需要准备好序列数据并提取特征。

1. **解压数据**：
   ```bash
   cd /home/SCS2026004/ckf_workspace2/code
   unzip 抗血栓活性肽fasta.zip -d final_raw_data
   ```
2. **生成标签与基础特征**：
   运行脚本以生成 `pos_labels.npy` 和 `neg_labels.npy`：
   ```bash
   python extract_classical_features.py
   ```
3. **提取 ESM-2 深度学习特征**：
   ```bash
   python extract_esm2_features.py
   ```
   *注意：此步骤会自动下载 ESM-2 权重，请确保网络通畅。*

---

## 3. 模型训练与评估

如果你需要重新训练模型或进行交叉验证：

1. **开始 5 折交叉验证训练**：
   ```bash
   python train.py
   ```
   训练好的模型权重将保存在 `checkpoints/` 目录下。
2. **模型评估与绘图**：
   ```bash
   python eval_hybrid.py     # 评估混合模型性能
   python plot_results.py    # 生成训练曲线与混淆矩阵
   python plot_comparison.py # 生成与传统模型对比图
   ```

---

## 4. 运行 Web 预测系统 (推荐)

项目提供了一个基于 Gradio 的图形化界面，支持单序列预测及注意力机制可视化。

```bash
python app.py
```
- **访问方式**：启动后，控制台会显示 `http://127.0.0.1:7860`。
- **功能说明**：输入多肽序列（如 Hirudin 序列），系统将输出活性概率、关键残基热图及位点分析报告。

---

## 5. 核心脚本说明

| 脚本名称 | 功能描述 |
| :--- | :--- |
| `app.py` | **核心入口**：启动 Web 交互界面 |
| `train.py` | 训练脚本：支持 5 折交叉验证与早停机制 |
| `models.py` | 模型架构：定义 CNN-BiLSTM-Attention 混合网络 |
| `extract_esm2_features.py` | 特征工程：利用 ESM-2 预训练模型提取特征 |
| `analyze_signal_peptide.py` | 数据处理：利用 SignalP 6.0 提取成熟肽序列 |
| `interpret_model.py` | 模型解释：生成残基贡献度分析报告 |

---

## 6. 常见问题 (FAQ)

- **显存不足 (OOM)**：如果运行 `app.py` 报错，请尝试在终端执行 `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`。
- **路径找不到**：代码中使用了部分绝对路径，请确保在 `/home/SCS2026004/ckf_workspace2/code` 目录下运行脚本。
- **SignalP 6.0 报错**：请确保 `tools/signalp6` 目录下的环境已正确安装（通常需要单独的 license）。

---
**项目维护**：成凯峰  
**最后更新**：2026-04-23
