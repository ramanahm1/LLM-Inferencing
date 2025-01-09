# Citation: https://www.youtube.com/watch?v=a_HHryXoDjM
import transformers
import torch
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device.upper()}")

model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

experiments = {
    "exp_1": {"temperature": 0.5, "top_k": 10, "top_p": 0.9, "repetition_penalty": 1.0, "max_length": 100},
    "exp_2": {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.0, "max_length": 100},
    "exp_3": {"temperature": 1.0, "top_k": 0, "top_p": 0.8, "repetition_penalty": 1.2, "max_length": 200},
    "exp_4": {"temperature": 1.2, "top_k": 100, "top_p": 1.0, "repetition_penalty": 1.5, "max_length": 300},
    "exp_5": {"temperature": 0.7, "top_k": 10, "top_p": 0.7, "repetition_penalty": 1.0, "max_length": 150},
}

results = {}

for exp_id, params in experiments.items():
    print(f"\nRunning {exp_id} with params: {params}")
    
    sequences = pipeline(
        "Hey how are you doing today?",
        return_full_text=False,
        max_length=params["max_length"],
        num_return_sequences=1,
        top_k=params["top_k"],
        top_p=params["top_p"],
        temperature=params["temperature"],
        repetition_penalty=params["repetition_penalty"],
        do_sample=True,
        truncation=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    results[exp_id] = [seq["generated_text"] for seq in sequences]

for exp_id, output in results.items():
    print(f"\n{exp_id} Output: {output}")
