import json
import os


from jinja2 import Environment
from tokenizers import Tokenizer as BaseTokenizer, Encoding
from huggingface_hub import snapshot_download
import torch


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    eos_token: str
    eos_token_id: int
    pad_token: str
    pad_token_id: int

    def __init__(self, tokenizer_path: str, tokenizer_config_path: str):
        super().__init__()
        with open(tokenizer_config_path, "r") as f:
            self.tokenizer_config = json.load(f)
        self.tokenizer = BaseTokenizer.from_file(tokenizer_path)
        self.chat_template = Environment().from_string(self.tokenizer_config["chat_template"])
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        if "pad_token" in self.tokenizer_config:
            self.pad_token = self.tokenizer_config["pad_token"]
            self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        else:
            self.pad_token = self.eos_token
            self.pad_token_id = self.eos_token_id

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode(self, text: str) -> Encoding:
        return self.encode_batch([text])[0]

    def encode_batch(self, text: str) -> list[Encoding]:
        return self.tokenizer.encode_batch_fast(text)

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens=False,
        skip_padding=True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_padding:
            token_ids = [i for i in token_ids if i != self.pad_token_id]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encodings_to_tensor(self, encodings: list[Encoding]) -> torch.Tensor:
        max_n = max([len(encoding.ids) for encoding in encodings])
        token_ids = torch.full(
            (len(encodings), max_n), fill_value=self.pad_token_id, dtype=torch.long
        )

        for i, encoding in enumerate(encodings):
            token_ids[i, : len(encoding.ids)] = torch.tensor(encoding.ids)

        return token_ids

    @classmethod
    def from_huggingface(cls, model_name: str):
        repo_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=["tokenizer.json", "tokenizer_config.json"],
        )
        return cls(
            tokenizer_path=os.path.join(repo_path, "tokenizer.json"),
            tokenizer_config_path=os.path.join(repo_path, "tokenizer_config.json"),
        )
