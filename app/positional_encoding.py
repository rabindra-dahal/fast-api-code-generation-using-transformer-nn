import torch
import torch.nn as nn
import math

class SimplePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        # 1. Create a matrix of zeros filled with the shape: (max_len x d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 2. Generate a column tensor representing positions [0, 1, 2, 3...]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Apply the Transformer wave math formula (using sine and cosine)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply Sine to even matrix columns
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply Cosine to odd matrix columns
        
        # Save this position matrix so PyTorch tracks it, but doesn't try to train it
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape is (1 sequence, number of words, coordinates dimension)
        # We slice our wave matrix to match the exact number of words we passed in
        return x + self.pe[:x.size(1), :]

# ==========================================
# TESTING IT WITH OUR WORD EMBEDDINGS
# ==========================================
# Let's reuse our 4 tokens from earlier: [from, fastapi, import, app]
# Shape: 1 sentence, 4 words, 4 dimensions each
mock_embeddings = torch.randn(1, 4, 4) 

print("--- ORIGINAL WORD EMBEDDINGS (No Order) ---")
print(mock_embeddings[0])

# Initialize our positional engine
pos_encoder = SimplePositionalEncoding(d_model=4, max_len=10)

# Add the position signals directly to our vectors!
ordered_embeddings = pos_encoder(mock_embeddings)

print("\n--- EMBEDDINGS AFTER POSITION ENCODING (Ordered) ---")
print(ordered_embeddings[0])
