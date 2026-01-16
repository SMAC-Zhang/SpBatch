import time
import torch
from vllm import LLM, SamplingParams

def benchmark_vllm():
    torch.cuda.set_device(0)
    
    model_name = "/data/zzx/Bamboo-DPO-v0_1"
    
    print(f"Loading model {model_name} on GPU 0...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True, 
        max_model_len=2048,
        dtype="bfloat16",
    )
    print(f"Model {model_name} loaded on GPU 0!")
    
    prompt_length = 8 
    num_tokens_to_generate = 8
    batch_sizes = [128, 64, 32, 16, 8]
    num_runs = 10
    
    base_prompt = "Hello world! How are you doing today? I hope everything is going well."
    
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Tokens to generate per run: {num_tokens_to_generate} tokens")
    print(f"Number of runs per batch size: {num_runs}\n")
    

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=num_tokens_to_generate,
    )
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"test batch size: {batch_size}")
        print(f"{'='*60}")
        
        prompts = [base_prompt] * batch_size
        
        # warm up
        if batch_size == batch_sizes[0]:
            print("warming up...")
            _ = llm.generate(prompts[:1], sampling_params)
            torch.cuda.synchronize()  # wait for GPU operations to complete
        
        # record token generation speed for each run
        speeds = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            
            outputs = llm.generate(prompts, sampling_params)
            
            torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            
            total_generated_tokens = 0
            for output in outputs:
                total_generated_tokens += len(output.outputs[0].token_ids)
            
            tokens_per_second = total_generated_tokens / elapsed_time
            
            speeds.append(tokens_per_second)
            
            print(f"Run {run+1:2d}: Time={elapsed_time:.3f}s, "
                  f"Generated tokens={total_generated_tokens}, "
                  f"Speed={tokens_per_second:.2f} token/s")
        
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        min_speed = min(speeds)

        with open("vllm.csv", "a+") as log_file:
            log_file.write(f"{max_speed:.6f},")
        
        print(f"\nBatch size {batch_size} statistics:")
        print(f"  Average speed: {avg_speed:.2f} token/s")
        print(f"  Max speed: {max_speed:.2f} token/s")
        print(f"  Min speed: {min_speed:.2f} token/s")
        print(f"  Speed range: {max_speed - min_speed:.2f} token/s")
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: No GPU detected!")
        exit(1)
    
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        benchmark_vllm()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()