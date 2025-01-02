## minrl

Reproduction of [GRPO](https://arxiv.org/abs/2402.03300), intended to be educational and efficient:

- On an RTX 3090, we start with a base model (Qwen3-1.7B-Base) and train on the [countdown task](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4), reaching 60% accuracy in under 3 hours.
- We incorporate improvements from [Dr. GRPO](https://arxiv.org/abs/2503.20783) and [DAPO](https://arxiv.org/abs/2503.14476v1), namely averaging over tokens rather than over batches and dropping reward scaling.
- We use LoRA to fit model training into 24GB of GPU memory.