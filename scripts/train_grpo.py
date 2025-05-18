import torch
import wandb

from minrl.modeling.lora import LoRAEmbedding, LoRALinear
from minrl.model import LanguageModel, to_compiled
from minrl.tasks.countdown import CountdownTask
from minrl.tokenizer import Tokenizer
from minrl.trainer import Trainer


def train(
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    batch_size: int = 160,
    num_answers_per_question: int = 16,
    gradient_batch_size: int = 1,
    lora_rank: int = 64,
    generation_length: int = 512,
):
    num_questions_per_batch = batch_size // num_answers_per_question
    tokenizer = Tokenizer.from_huggingface(model_name)
    task = CountdownTask(tokenizer, batch_size=num_questions_per_batch)
    max_seq_len = generation_length + 200
    model = (
        LanguageModel.from_hf(
            model_name, max_seq_len=max_seq_len, dtype=torch.float16, device="cuda"
        )
        .requires_grad_(False)
        .init_kv_cache(batch_size, seq_len=max_seq_len)
    )

    # Keep lora weights in float32
    model = LoRAEmbedding.find_and_replace(model, rank=lora_rank, dtype=torch.float32)
    model = LoRALinear.find_and_replace(
        model,
        pattern=r"(self_attn\.(q|k|v|o)_proj)|(mlp\.(gate|down|up)_proj)",
        rank=lora_rank,
        dtype=torch.float32,
    )
    model = to_compiled(model)

    wandb.init(project="minrl")

    trainer = Trainer(
        model=model,
        task=task,
        lr=1e-5,
        max_grad_norm=1.0,
        num_answers_per_question=num_answers_per_question,
        generation_length=generation_length,
        gradient_batch_size=gradient_batch_size,
        ignore_incomplete_responses=False,
        max_steps=1000,
    )

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        trainer.run_training()


if __name__ == "__main__":
    train()
