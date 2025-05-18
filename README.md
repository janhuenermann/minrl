## minrl

Reproduction of [GRPO](https://arxiv.org/abs/2402.03300), intended to be educational and efficient:

- On an RTX 3090, we start with a base model (Qwen3-1.7B-Base) and train on the [countdown task](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4), reaching 60% accuracy in under 3 hours.
- We incorporate improvements from [Dr. GRPO](https://arxiv.org/abs/2503.20783) and [DAPO](https://arxiv.org/abs/2503.14476v1), namely averaging over tokens rather than over batches and dropping reward scaling.
- We use LoRA to fit model training into 24GB of GPU memory.

### Setup

In order to set up a training environment, you need to install the dependencies, including `flash-attn`.

```bash
# Clone the repository
git clone git@github.com:janhuenermann/minrl.git && cd minrl

# It is recommended to create a new virtual environment
python -m venv .env
source .env/bin/activate

# Install package and core dependencies
pip install -e .

# Install flash-attn with ninja for faster build times (this may take a while)
pip install ninja
MAX_JOBS=4 pip install -v --no-build-isolation flash-attn
```

### Training

To start training, you can simply run the `scripts/train_grpo.py` scrip, which will automatically download the required artifacts (dataset and base model) and start training.

```bash
python scripts/train_grpo.py
```