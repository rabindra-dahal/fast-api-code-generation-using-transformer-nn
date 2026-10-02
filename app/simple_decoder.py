import time

# 1. The Decoder's "Brain" (Learned Word Associations)
# It looks at the current word to decide what word should come next.
decoder_brain = {
    "<sos>": "from",       # <sos> means "Start of Sequence"
    "from": "fastapi",
    "fastapi": "import",
    "import": "FastAPI\n",
    "FastAPI\n": "app",
    "app": "=",
    "=": "FastAPI()",
    "FastAPI()": "<eos>"   # <eos> means "End of Sequence" (Stop writing!)
}

# 2. The Auto-Regressive Loop (Guessing word by word)
current_word = "<sos>"
generated_code = []

print("Decoder is generating code...")
while current_word != "<eos>":
    # The decoder looks up the next word based on what it just wrote
    next_word = decoder_brain[current_word]
    print(f"Current token: {current_word} -> Next token: {next_word}")
    
    if next_word != "<eos>":
        generated_code.append(next_word)
        print(f"Generated token: {next_word}")
        time.sleep(0.4) # Simulating "thinking" time
        
    current_word = next_word

# 3. Output the final assembled code
print("\n--- Final Generated Output ---")
print(" ".join(generated_code))
