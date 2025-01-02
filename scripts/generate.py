import torch

from minrl.model import LanguageModel, generate
from minrl.tokenizer import Tokenizer


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

batch_size = 1
device = torch.device("cuda:0")
model_name = "Qwen/Qwen3-1.7B"

model = LanguageModel.from_hf(model_name, device=device, dtype=torch.float16)
model = model.init_kv_cache(batch_size, seq_len=1024)
tokenizer = Tokenizer.from_huggingface(model_name)

while True:
    text = input("> ")
    if text == "exit":
        break

    conversation = tokenizer.encode_chat([{"role": "user", "content": text}])

    input_ids = tokenizer.tokenize(conversation).ids
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.int64)[None]
    output_ids = generate(
        model,
        input_ids,
        generation_length=800,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.detokenize(output_ids[0].tolist(), skip_special_tokens=True)
    print(response)
