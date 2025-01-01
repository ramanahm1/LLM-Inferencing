# Citation: https://www.youtube.com/watch?v=a_HHryXoDjM
import transformers
import torch
from transformers import AutoTokenizer

# Checking for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device.upper()}")

# Model init
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Pipeline init
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

# Run the inference pipeline
sequences = pipeline(
    "Hey how are you doing today?",
    return_full_text=False,
    max_length=400,
    num_return_sequences=1,
    top_k=10,
    do_sample=True,
    truncation=False,
    eos_token_id=tokenizer.eos_token_id,
)

# Check output
for seq in sequences:
    print(seq["generated_text"])