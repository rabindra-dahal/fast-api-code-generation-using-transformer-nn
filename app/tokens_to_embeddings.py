import torch
import torch.nn as nn

# 1. Re-use the setup from our previous step
vocabulary = ["<pad>", "<sos>", "<eos>", "<unk>", "from", "fastapi", "import", "FastAPI", "app", "=", "()"]
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

# Tokenized input tensor for: "from fastapi import app"
# Mapping matches indices: [4, 5, 6, 8]
input_tokens = torch.tensor([4, 5, 6, 8], dtype=torch.long)

# 2. Define the Embedding Layer
VOCAB_SIZE = len(vocabulary)  # 11 unique words
EMBEDDING_DIM = 3            # Each word will turn into 3 spatial coordinate points (e.g., [x, y, z])

# nn.Embedding is a lookup matrix of size (VOCAB_SIZE x EMBEDDING_DIM)
embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

# 3. Pass the integer tokens through the embedding matrix
word_vectors = embedding_layer(input_tokens)

print("--- TOKEN INPUT ---")
print(f"Token Indices: {input_tokens.tolist()}")
print(f"Input Shape:   {input_tokens.shape} (4 words)\n")

print("--- NEURAL SPATIAL EMBEDDINGS VECTOR ---")
print(word_vectors)
print(f"\nEmbedding Output Shape: {word_vectors.shape} (4 words, each with 3 coordinates)")
