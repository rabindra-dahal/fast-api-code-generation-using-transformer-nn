import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# =======================================================
# 1. TWO-WAY TOKENIZER SETUP (ENGLISH & PY)
# =======================================================
english_vocab = ["<pad>", "<unk>", "create", "a", "simple", "fastapi", "app", "build", "web", "application"]
python_vocab  = ["<pad>", "<sos>", "<eos>", "<unk>", "fastapi", "from", "import", "FastAPI", "app", "=", "FastAPI()"]

en_w2i = {w: i for i, w in enumerate(english_vocab)}
en_i2w = {i: w for i, w in enumerate(english_vocab)}
py_w2i = {w: i for i, w in enumerate(python_vocab)}
py_i2w = {i: w for i, w in enumerate(python_vocab)}

# # Dynamic Dataset: Pairing English input sentences with targeted FastAPI outputs
# dataset = [
#     ("create a simple fastapi app", ["<sos>", "from", "fastapi", "import", "FastAPI", "app", "=", "FastAPI()", "<eos>"]),
#     ("build a web application",     ["<sos>", "from", "fastapi", "import", "FastAPI", "app", "=", "FastAPI()", "<eos>"])
# ]

import os

# 1. Dynamically read the dataset from our external file
dataset = []

if os.path.exists("data.txt"):
    with open("data.txt", "r") as file:
        for line in file:
            if "|" in line:
                english_part, python_part = line.strip().split("|")
                
                # Tokenize the Python string automatically and add special tokens
                python_tokens = ["<sos>"] + python_part.split() + ["<eos>"]
                
                # Append the parsed pair to our dataset list
                dataset.append((english_part, python_tokens))
    print(f" Successfully loaded {len(dataset)} examples from data.txt dynamically!")
else:
    print(" data.txt not found! Please create the file first.")


# Numerical transformation helper
def pad_seq(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq))

max_en_len = max(len(p[0].split()) for p in dataset)
max_py_len = max(len(p[1]) for p in dataset)

# Construct matching parallel tensor matrices
src_list, tgt_list = [], []
for en_phrase, py_tokens in dataset:
    src_list.append(pad_seq([en_w2i.get(w, 1) for w in en_phrase.split()], max_en_len, en_w2i["<pad>"]))
    tgt_list.append([py_w2i[w] for w in py_tokens])

src_tensor = torch.tensor(src_list, dtype=torch.long)
tgt_tensor = torch.tensor(tgt_list, dtype=torch.long)

# =======================================================
# 2. TRANSFORMER CORE ARCHITECTURE
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
        return x + self.pe[:x.size(1), :]

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=32):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # PyTorch built-in sequence transformer infrastructure block
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=64, batch_first=True
        )
        self.classifier = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))
        
        # Create look-ahead safety mask for decoder targets
        sz = tgt.size(1)
        tgt_mask = torch.triu(torch.full((sz, sz), float('-inf'), device=tgt.device), diagonal=1)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.classifier(out)

# =======================================================
# 3. TRAINING MODE
# =======================================================
model = EncoderDecoderTransformer(len(english_vocab), len(python_vocab))
criterion = nn.CrossEntropyLoss(ignore_index=py_w2i["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print(" Training full Encoder-Decoder model to link English -> Python...")
model.train()
for epoch in range(250):
    optimizer.zero_grad()
    
    # Decoder inputs ignore the final <eos> token, expected targets are shifted by 1 index
    tgt_in = tgt_tensor[:, :-1]
    tgt_out = tgt_tensor[:, 1:]
    
    predictions = model(src_tensor, tgt_in)
    loss = criterion(predictions.reshape(-1, len(python_vocab)), tgt_out.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/250 | System Loss: {loss.item():.6f}")

print("\nModel trained successfully!")
print("-" * 65)

# =======================================================
# 4. AUTO-REGRESSIVE INFERENCE WITH ENGLISH PROMPTS
# =======================================================
def generate_code_from_prompt(english_prompt):
    model.eval()
    
    # Tokenize and pad the user prompt
    prompt_tokens = [en_w2i.get(w, en_w2i["<unk>"]) for w in english_prompt.lower().split()]
    prompt_padded = pad_seq(prompt_tokens, max_en_len, en_w2i["<pad>"])
    src_in = torch.tensor([prompt_padded], dtype=torch.long)
    
    # Seed the decoder sequence
    generated_py = [py_w2i["<sos>"]]
    
    for step in range(12):
        tgt_in = torch.tensor([generated_py], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(src_in, tgt_in)
            
        next_token_idx = torch.argmax(logits[0, -1, :]).item()
        
        if next_token_idx == py_w2i["<eos>"]:
            break
            
        generated_py.append(next_token_idx)
        
    final_tokens = [py_i2w[idx] for idx in generated_py if idx != py_w2i["<sos>"]]
    return " ".join(final_tokens).replace("app = FastAPI ()", "app = FastAPI()")

# Test run using one of our prompts
my_prompt = "create a simple fastapi app"
print(f" Input English Prompt: '{my_prompt}'")
compiled_code = generate_code_from_prompt(my_prompt)
print(f" AI Generated Code:   {compiled_code}")
print("-" * 65)
