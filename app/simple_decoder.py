import time

# 1. The Decoder's "Brain"
decoder_brain = {
    "<sos>": "from",
    "from": "fastapi",
    "fastapi": "import",
    "import": "FastAPI\n",
    "FastAPI\n": "app",
    "app": "=",
    "=": "FastAPI()",
    "FastAPI()": "<eos>"
}

current_word = "<sos>"
generated_code = []
MAX_TOKENS = 10  # Protection limit: never generate more than 10 words

print("Decoder is generating code using a safe FOR loop...")

for step in range(MAX_TOKENS):
    next_word = decoder_brain[current_word]
    
    # If the AI naturally decides to stop, break out of the loop early
    if next_word == "<eos>":
        print(f"Step {step+1}: Hit <eos>! Stopping naturally.")
        break
        
    generated_code.append(next_word)
    print(f"Step {step+1}: Generated token -> {next_word}")
    time.sleep(0.4)
    
    # Update our position for the next iteration of the loop
    current_word = next_word

# 3. Output the final assembled code
print("\n--- Final Generated Output ---")
print(" ".join(generated_code))
