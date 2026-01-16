import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

model_name = "/data/zzx/Bamboo-DPO-v0_1"

print(f"Loading model: {model_name}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0" 
)

device = torch.device("cuda:0")
model = model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded on device: {device}")
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 50)

prompt_length = 8
max_new_tokens = 8
num_runs = 20
batch_sizes = [128, 64, 32, 16, 8]

# Warm up GPU (run once to avoid initialization overhead)
print("Warming up GPU...")
with torch.no_grad():
    test_input = torch.randint(0, tokenizer.vocab_size, (1, prompt_length)).to(device)
    _ = model.generate(test_input, max_new_tokens=2, do_sample=False)

# Test different batch sizes
for batch_size in batch_sizes:
    print(f"\nTesting batch size: {batch_size}")
    
    input_ids = torch.randint(
        0, tokenizer.vocab_size, 
        (batch_size, prompt_length)
    ).to(device)
    
    speeds = []
    for run in range(num_runs):
        torch.cuda.empty_cache()

        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_tokens = batch_size * max_new_tokens
        elapsed_time = end_time - start_time
        speed = total_tokens / elapsed_time
        
        speeds.append(speed)
        
        print(f"  Run {run+1:2d}: {speed:.2f} tokens/s")
    
    # Calculate statistics
    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    min_speed = min(speeds)

    with open("transformer.csv", "a+") as log_file:
        log_file.write(f"{max_speed:.6f},")
    
    print(f"  Average speed: {avg_speed:.2f} tokens/s")
    print(f"  Max speed: {max_speed:.2f} tokens/s")
    print(f"  Min speed: {min_speed:.2f} tokens/s")

print("\n" + "="*50)
print("Testing completed!")
print(f"GPU memory usage: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")