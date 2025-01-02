import torch

from minrl.model import LanguageModel, generate, to_compiled
from minrl.modeling.lora import LoRAEmbedding, LoRALinear
from minrl.tokenizer import Tokenizer
from minrl.tasks.countdown import CountdownTask


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

batch_size = 160
device = torch.device("cuda:0")
model_name = "Qwen/Qwen3-1.7B"

model = LanguageModel.from_hf(model_name, device=device, dtype=torch.float16)
model = model.init_kv_cache(batch_size, seq_len=1024)
model = to_compiled(model)

tokenizer = Tokenizer.from_huggingface(model_name)
task = CountdownTask(tokenizer, batch_size=batch_size)
data_loader = task.train_data_loader()


def benchmark(input_ids, n_tokens: int):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    output_ids = generate(
        model,
        input_ids,
        generation_length=n_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False,
    )
    stop_event.record()

    torch.cuda.synchronize()

    n_output_tokens = (output_ids != tokenizer.pad_token_id).sum().item()
    n_input_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
    n_generated_tokens = n_output_tokens - n_input_tokens

    execution_time = start_event.elapsed_time(stop_event)
    time_per_token = execution_time / n_generated_tokens
    tokens_per_sec = 1000 / time_per_token

    print(tokenizer.decode(output_ids[0].cpu().numpy()))
    print("========================")
    print(
        f"Tokens: {n_input_tokens} input, {n_output_tokens} output, {n_generated_tokens} generated"
    )
    print(f"Execution time: {execution_time} ms")
    print(f"Tokens per second: {tokens_per_sec:.3f} tokens/sec ({time_per_token:.3f} ms per token)")
    print("========================")


for batch, _ in zip(data_loader, range(10)):
    input_ids = torch.as_tensor(batch["prefix_token_ids"]).to(device)
    benchmark(input_ids, n_tokens=512)
