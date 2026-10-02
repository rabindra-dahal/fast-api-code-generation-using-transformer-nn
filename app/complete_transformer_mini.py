import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ==========================================
# 1. TOKENIZER & CORPUS SETUP
# ==========================================
# Our vocabulary of unique FastAPI structural tokens
vocabulary = ["<pad>", "<sos>", "<eos>", "<unk>", "from", "fastapi", "import", "FastAPI", "app", "=", "()"]
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}

# The exact "brain dictionary" our model will use to simulate learned weights
decoder_brain = {
    word_to_idx["<sos>"]: word_to_idx["from"],
    word_to_idx["from"]: word_to_idx["fastapi"],
    word_to_idx["fastapi"]: word_to_idx["import"],
    word_to_idx["import"]: word_to_idx["FastAPI"],
    word_to_idx["FastAPI"]: word_to_idx["app"],
    word_to_idx["app"]: word_to_idx["="],
    word_to_idx["="]: word_to_idx["()"],
    word_to_idx["()"]: word_to_idx["<eos>"]
}

# ==========================================
# 2. POSITION ENCODING LAYER BLOCK
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adds order coordinates directly to the word embeddings
        return x + self.pe[:x.size(1), :]

# ==========================================
# 3. MANUAL CAUSAL ATTENTION LAYER BLOCK
# ==========================================
class CustomCausalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Linear projections for Query, Key, and Value roles
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        sz = x.size(1) # Get current sequence length
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Step A: Raw Attention Matrix Multiplication (Q @ K transposed)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Step B: Apply the Upper Triangular Blinding Mask (-inf to future words)
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x.device), diagonal=1)
        masked_scores = raw_scores + mask
        
        # Step C: Convert to exact probabilities (Softmax forces -inf to become 0.0%)
        attention_matrix = F.softmax(masked_scores, dim=-1)
        
        # Step D: Multiply probabilities by raw Values to get contextualized coordinates
        return torch.matmul(attention_matrix, V), attention_matrix

# ==========================================
# 4. RUNNING THE AUTO-REGRESSIVE GENERATION
# ==========================================
# Setting up our dimensions
VOCAB_SIZE = len(vocabulary)
EMBEDDING_DIM = 4  # 4 spatial dimensions per word

# Instantiate our components
embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
pos_encoder = PositionalEncoding(d_model=EMBEDDING_DIM)
attention_block = CustomCausalAttention(d_model=EMBEDDING_DIM)

# Let's start the compilation!
current_tokens = [word_to_idx["<sos>"]]
MAX_TOKENS = 12

print(" Starting pure Transformer inference pipeline...")
print("Initial Input Sequence:", [idx_to_word[t] for t in current_tokens])
print("-" * 50)

for step in range(MAX_TOKENS):
    # 1. Convert our current tokens array into a PyTorch long tensor matrix
    input_tensor = torch.tensor([current_tokens], dtype=torch.long)
    
    # 2. Pipeline processing: Embeddings -> Order Coding -> Self-Attention Masking
    embeddings = embedding_layer(input_tensor)
    ordered_embeddings = pos_encoder(embeddings)
    context_vectors, final_attn_matrix = attention_block(ordered_embeddings)
    
    # 3. Predict the next token using our simulated model brain
    last_token_idx = current_tokens[-1]
    next_token_idx = decoder_brain.get(last_token_idx, word_to_idx["<unk>"])
    
    if next_token_idx == word_to_idx["<eos>"]:
        print(f"\n Step {step+1}: Hit <eos> marker! Safe exit triggered.")
        break
        
    current_tokens.append(next_token_idx)
    print(f"Step {step+1}: Generated -> '{idx_to_word[next_token_idx]}'")
    time.sleep(0.3)

# ==========================================
# 5. THE FINAL GENERATED CODE OUTPUT
# ==========================================
final_code_words = [idx_to_word[t] for t in current_tokens if t not in [word_to_idx["<sos>"], word_to_idx["<eos>"]]]
assembled_code = " ".join(final_code_words)

# Clean up basic python structural formatting spacing
assembled_code = assembled_code.replace("app = FastAPI ()", "app = FastAPI()")

print("-" * 50)
print(" FINAL ASSEMBLED FASTAPI APPLICATION:")
print(assembled_code)
print("-" * 50)
print(" Last Attention Matrix Matrix Shape processed:", final_attn_matrix.shape)
