import torch

# 1. Our Tiny FastAPI Raw Text Corpus
data_corpus = [
    "from fastapi import FastAPI",
    "app = FastAPI()",
    "def root(): return hello"
]

# 2. Build the Vocabulary Dictionary (Unique Words List)
# We include special tokens that tell the AI how to behave structures.
vocabulary = ["<pad>", "<sos>", "<eos>", "<unk>"]

# Extract every unique word from our data corpus
for sentence in data_corpus:
    # Clean the sentence and split it by spaces
    words = sentence.replace("()", " ()").replace(":", " :").split()
    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)

print("Vocabulary : ", vocabulary)
# Create lookup tables: Word-to-Index (Integer) and Index-to-Word
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}

print("--- OUR EXTRACTED VOCABULARY DICTIONARY ---")
print(word_to_idx)
print(f"Total Unique Tokens: {len(vocabulary)}\n")

# 3. Tokenize a new prompt sentence into a PyTorch Integer Tensor
new_prompt = "from fastapi import app"
tokens = []

for word in new_prompt.split():
    # If the word isn't in our vocab lookup, fallback to the <unk> (unknown) token index
    idx = word_to_idx.get(word, word_to_idx["<unk>"])
    tokens.append(idx)

# Convert our standard Python array list into a heavy-duty PyTorch Int Tensor
torch_tensor = torch.tensor(tokens, dtype=torch.long)

print("--- TOKENIZATION RESULT ---")
print(f"Original Input Phrase: '{new_prompt}'")
print(f"PyTorch Integer Tensor Output: {torch_tensor}")
print(f"Tensor Shape: {torch_tensor.shape} (1 row containing 4 token indices)")
