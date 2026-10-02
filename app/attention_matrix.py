import torch
import torch.nn.functional as F

# 1. Mock inputs for 3 tokens: ["from", "fastapi", "import"]
# Shape: (1 sentence, 3 words, 4 dimensions per word)
torch.manual_seed(42) # Lock random weights for clean output visualization
ordered_embeddings = torch.randn(1, 3, 4)

# 2. Linear projection layers to create Q, K, and V matrices
# Each layer converts a word embedding into its distinct attention role
query_layer = torch.nn.Linear(4, 4, bias=False)
key_layer   = torch.nn.Linear(4, 4, bias=False)
value_layer = torch.nn.Linear(4, 4, bias=False)

# Compute Q, K, and V matrices
Q = query_layer(ordered_embeddings) # What each word asks
K = key_layer(ordered_embeddings)   # What each word answers
V = value_layer(ordered_embeddings) # The real content

# 3. COMPUTE THE ATTENTION MATRIX (Q multiplied by Transposed K)
# We multiply the query of word X with the key of word Y to find alignment
# Shape results in a clean (3x3) interaction grid
attention_scores = torch.matmul(Q, K.transpose(-2, -1))

# 4. SCALE AND APPLY SOFTMAX (Turn raw math scores into % probabilities)
d_k = Q.shape[-1] # Look up dimensions (4)
scaled_scores = attention_scores / (d_k ** 0.5) # Prevent massive matrix saturation
attention_matrix = F.softmax(scaled_scores, dim=-1) # Force each row to total 100%

print("--- THE 3x3 ATTENTION MATRIX MAP ---")
print("Columns/Rows represent tokens: [from, fastapi, import]")
print(attention_matrix[0].detach().numpy())

# 5. GENERATE FINAL CONTEXT VECTOR (Multiply weights matrix by raw Content Values)
contextual_output = torch.matmul(attention_matrix, V)
print("\n--- FINAL CONTEXTUALIZED OUTPUT SHAPE ---")
print(contextual_output.shape)
