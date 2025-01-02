import re
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download

from minrl.tokenizer import Tokenizer

USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /). Each number should be used exactly once.\n"
    "Return the final answer in <answer></answer> tags, for example `<answer>(1 + 2) / 3</answer>`\n"
    "To find a solution, you can keep a scratchpad of reasoning steps in <think>...</think> tags.\n"
)
RESPONSE_PROMPT = "<think>\nLet me think step by step."


class CountdownTask:
    def __init__(self, tokenizer: Tokenizer, batch_size: int, test_batch_size: int = 128):
        data_path = snapshot_download(
            repo_id="Jiayi-Pan/Countdown-Tasks-3to4",
            allow_patterns=["*.parquet"],
            repo_type="dataset",
        )
        data = pd.read_parquet(os.path.join(data_path, "data"))

        self.train_data = CountdownTasksDataset(data.iloc[:-100], tokenizer)
        self.test_data = CountdownTasksDataset(data.iloc[-100:], tokenizer)
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

    def train_data_loader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def test_data_loader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def compute_reward(self, response: str, batch: dict, index: int):
        # Strip end token if present
        if self.tokenizer.eos_token and response.endswith(self.tokenizer.eos_token):
            response = response[: -len(self.tokenizer.eos_token)]

        answer_reward = get_answer_reward(response, batch["numbers"][index], batch["target"][index])
        format_reward = get_format_reward("<think>" + response)
        total_reward = answer_reward + 0.1 * format_reward
        return {"reward": total_reward}


class CountdownTasksDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: Tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitems__(self, indices: list[int]):
        rows = self.data.iloc[indices]
        chats = []

        for _, row in rows.iterrows():
            numbers = [int(n) for n in row["nums"]]
            target = int(row["target"])
            question = USER_TEMPLATE.format(numbers=numbers, target=target)
            chats.append(
                self.tokenizer.format_chat([{"role": "user", "content": question}])
                + RESPONSE_PROMPT
            )

        tokenized_chats = self.tokenizer.encode_batch(chats)
        prefix_token_ids = self.tokenizer.encodings_to_tensor(tokenized_chats)
        return {
            "target": [int(row["target"]) for _, row in rows.iterrows()],
            "numbers": [[int(n) for n in row["nums"]] for _, row in rows.iterrows()],
            "prefix_token_ids": prefix_token_ids,
        }


def get_format_reward(response: str) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Negative lookahead to ensure <think> and <answer> tags are not nested
    content_regex = r"(?:(?!</?think|answer>).)*"
    format_regex = (
        r"^<think>(?P<think>" + content_regex + r")<\/think>\s*"
        r"<answer>(?P<answer>" + content_regex + r")<\/answer>$"
    )

    if re.match(format_regex, response, re.DOTALL):
        return 0.0

    return -1.0


def get_answer_reward(response: str, numbers: list[int], target: int) -> float:
    answer_regex = r"<answer>(.*?)<\/answer>\s*$"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0
