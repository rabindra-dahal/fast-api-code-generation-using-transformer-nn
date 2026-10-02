import torch
import torch.nn.functional as F

# 1. Let's recreate our raw 3x3 Attention Scores (Before turning into percentages)
# This represents: [from, fastapi, import]
raw_scores = torch.tensor([
    [2.1, 1.5, 0.3],  # Row for 'from'
    [0.8, 3.4, 1.1],  # Row for 'fastapi'
    [1.2, 0.5, 2.9]   # Row for 'import'
])

# 2. CREATE THE MASK MATRIX (Upper Triangle of Negative Infinities)
seq_len = raw_scores.size(0)
# torch.triu generates an upper triangular matrix
# diagonal=1 ensures we leave the current word alone but block everything to its right
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

print("--- THE BLINDING MASK MATRIX ---")
print(mask)

# 3. APPLY THE MASK (Add it directly to our raw attention scores)
masked_scores = raw_scores + mask

print("\n--- MASKED ATTENTION SCORES ---")
print(masked_scores)

# 4. CONVERT TO PERCENTAGES (Softmax)
# -inf turns into exactly 0.0%
attention_matrix = F.softmax(masked_scores, dim=-1)

print("\n--- THE FINAL SAFE DECODER ATTENTION MATRIX MAP ---")
print("Columns/Rows: [from, fastapi, import]")
print(attention_matrix.numpy())
