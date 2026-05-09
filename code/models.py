import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        weights = self.attention(x) # (batch_size, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum
        context = torch.sum(weights * x, dim=1) # (batch_size, hidden_dim)
        return context, weights

class HybridModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, kernel_size=3, lstm_hidden=128, num_layers=2, dropout=0.3):
        super(HybridModel, self).__init__()
        
        # 1. CNN Module: Capture local motifs
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. BiLSTM Module: Capture long-range dependencies
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, 
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout)
        
        # 3. Attention Module: Focus on key regions
        self.attention = AttentionLayer(lstm_hidden * 2) # *2 for bidirectional
        
        # 4. Classification Module
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        
        # CNN expects (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        
        # LSTM expects (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, weights = self.attention(lstm_out)
        
        # Final classification
        out = self.classifier(attn_out)
        return out.squeeze(), weights

if __name__ == "__main__":
    # Test the model with dummy data
    # ESM-2 feature dim is 320 for the 8M model we used
    test_input = torch.randn(8, 100, 320) 
    model = HybridModel(input_dim=320)
    output, weights = model(test_input)
    print(f"Output shape: {output.shape}") # Should be (8,)
    print(f"Weights shape: {weights.shape}") # Should be (8, 100, 1)
    print("Model architecture test passed!")
