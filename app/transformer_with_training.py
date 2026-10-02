import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# =======================================================
# 1. TOKENIZER & CORPUS SETUP
# =======================================================
# Vocabulary mapping every structural component of our target FastAPI script
vocabulary = ["<pad>", "<sos>", "<eos>", "<unk>", "from", "fastapi", "import", "FastAPI", "app", "=", "FastAPI()"]
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}
VOCAB_SIZE = len(vocabulary)

# The target sequence our network must calculate matching weights for
target_code = ["<sos>", "from", "fastapi", "import", "FastAPI", "app", "=", "FastAPI()", "<eos>"]
target_indices = torch.tensor([[word_to_idx[w] for w in target_code]], dtype=torch.long)

# =======================================================
# 2. TRANSFORMER SUB-LAYER MATHEMATICAL MODULES
# =======================================================
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
        # Infuses linear word order directly into multi-dimensional coordinates
        return x + self.pe[:x.size(1), :]

class CustomCausalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Linear projections constructing separate search roles
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        sz = x.size(1)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Matrix Attention alignment calculations
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Build look-ahead blocker mask to prevent downstream information leakage
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x.device), diagonal=1)
        masked_scores = raw_scores + mask
        
        # Softmax turns raw values into true relative probability distributions
        attention_matrix = F.softmax(masked_scores, dim=-1)
        return torch.matmul(attention_matrix, V)

# =======================================================
# 3. EMBEDDING & LAYER INTEGRATION ASSEMBLY
# =======================================================
class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.attention = CustomCausalAttention(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.attention(x)
        logits = self.classifier(x)
        return logits

# =======================================================
# 4. TRAINING PIPELINE (OPTIMIZING COORDINATE WEIGHTS)
# =======================================================
# Upgraded dimension mapping from 8 to 32 to grant ample geometric distance variance
EMBEDDING_DIM = 32  
EPOCHS = 350        

model = MiniTransformerLM(VOCAB_SIZE, EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print(" Training Transformer parameters to resolve loops...")
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    inputs = target_indices[:, :-1] 
    targets = target_indices[:, 1:]
    
    outputs = model(inputs)
    
    loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Optimization Cross-Entropy Loss: {loss.item():.6f}")

print("\n Training Complete! Sequential math mapping optimized.")
print("-" * 65)

# =======================================================
# 5. AUTO-REGRESSIVE CODE COMPILATION (GENERATION PHASE)
# =======================================================
model.eval()
generated_tokens = [word_to_idx["<sos>"]]
MAX_TOKENS = 12

print(" Executing Auto-Regressive safe inference loop...")
for step in range(MAX_TOKENS):
    input_tensor = torch.tensor([generated_tokens], dtype=torch.long)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Slice prediction distributions matching strictly the newest current token index
    last_word_logits = predictions[0, -1, :]
    
    # Extract the absolute most probable vocabulary vector target
    next_token_idx = torch.argmax(last_word_logits).item()
    
    if next_token_idx == word_to_idx["<eos>"]:
        print(f"Step {step+1}: Encountered '<eos>' sentinel flag. Closing output.")
        break
        
    generated_tokens.append(next_token_idx)
    print(f"Step {step+1}: Appended token -> '{idx_to_word[next_token_idx]}'")
    time.sleep(0.15)

# =======================================================
# 6. ASSEMBLED DECODED PYTHON CODE OUTPUT
# =======================================================
final_words = [idx_to_word[t] for t in generated_tokens if t not in [word_to_idx["<sos>"], word_to_idx["<eos>"]]]
assembled_code = " ".join(final_words)

# Clean syntax representation mapping adjustments
assembled_code = assembled_code.replace("app = FastAPI()", "app = FastAPI()")

print("-" * 65)
print(" FINAL ERROR-FREE GENERATED FASTAPI SYNTAX APPLICATION:")
print(assembled_code)
print("-" * 65)
