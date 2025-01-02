from dataclasses import dataclass

import numpy as np
from torch import Tensor
import torch
from torch.nn import functional as F

from minrl.modeling.lora import lora_mode
from minrl.tokenizer import Tokenizer


@dataclass
class Episode:
    token_ids: Tensor
    prompt_ids: Tensor
    generated_ids: Tensor
    reward: float
    advantage: float | None = None
    response: str | None = None
    num_generated_tokens: int | None = None


def process_generations(
    task,
    batch,
    outputs,
    ignore_incomplete_responses: bool = True,
    normalize: bool = True,
) -> list[Episode]:
    tokenizer: Tokenizer = task.tokenizer
    token_ids = outputs["token_ids"].cpu()
    prompt_ids = outputs["prompt_ids"].cpu()

    generated_ids = token_ids.clone()
    generated_ids[:, : prompt_ids.size(1)] = torch.where(
        prompt_ids == tokenizer.pad_token_id,
        generated_ids[:, : prompt_ids.size(1)],
        tokenizer.pad_token_id,
    )

    episodes = []
    answers_per_question = generated_ids.size(0) // len(batch["target"])

    for i in range(len(batch["target"])):
        group_episodes = []
        for k in range(answers_per_question):
            m = i * answers_per_question + k
            response = tokenizer.decode(generated_ids[m])

            # Stop after first </answer> token and add EOS token
            # This is for base models that don't yet know how to use EOS tokens
            if "</answer>" in response and not response.endswith(tokenizer.eos_token):
                new_response = response.split("</answer>", 1)[0] + "</answer>" + tokenizer.eos_token
                new_ids = torch.as_tensor(tokenizer.encode(new_response).ids)
                prompt_length = (generated_ids[m] != tokenizer.pad_token_id).int().argmax()

                if prompt_length + len(new_ids) <= generated_ids.size(1):
                    response = new_response
                    token_ids[m, prompt_length : prompt_length + len(new_ids)] = new_ids
                    token_ids[m, prompt_length + len(new_ids) :] = tokenizer.pad_token_id
                    generated_ids[m, prompt_length : prompt_length + len(new_ids)] = new_ids
                    generated_ids[m, prompt_length + len(new_ids) :] = tokenizer.pad_token_id
                else:
                    # New response is too long - don't modify it
                    pass

            if ignore_incomplete_responses and not response.endswith(tokenizer.eos_token):
                continue  # Response is not complete

            num_response_tokens = (
                (generated_ids[m] != tokenizer.pad_token_id).count_nonzero().item()
            )

            # Compute the reward for given response
            reward_dict = task.compute_reward(response=response, batch=batch, index=i)
            group_episodes.append(
                Episode(
                    token_ids=token_ids[m],
                    prompt_ids=prompt_ids[m],
                    generated_ids=generated_ids[m],
                    reward=reward_dict["reward"],
                    response=response,
                    num_generated_tokens=num_response_tokens,
                )
            )

        if normalize:
            if len(group_episodes) < 2:
                continue  # Not enough answers for this question

            # Compute average reward
            avg_reward = np.mean([ep.reward for ep in group_episodes])
            for ep in group_episodes:
                ep.advantage = ep.reward - avg_reward

        episodes.extend(group_episodes)

    return episodes


def compute_loss(
    model,
    token_ids: Tensor,
    generated_ids: Tensor,
    advantages: Tensor,
    pad_token_id: int,
    beta: float = 0.0,
):
    targets = generated_ids[:, 1:]
    mask = targets != pad_token_id

    logits = model.forward(token_ids[:, :-1], return_logits=True)
    logp = compute_logp(logits, targets, pad_token_id)
    policy_loss = -logp * advantages[:, None]

    if beta > 0.0:
        # Forward reference policy
        with torch.no_grad(), lora_mode(model, enabled=False):
            ref_logits = model.forward(token_ids[:, :-1], return_logits=True)
            ref_logp = compute_logp(ref_logits, targets, pad_token_id)

        kl_loss = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
    else:
        kl_loss = torch.zeros_like(policy_loss)

    loss = (policy_loss + beta * kl_loss) * mask
    return loss.sum(), mask.sum()


def compute_logp(logits, targets, pad_token_id):
    return -F.cross_entropy(
        logits.float().flatten(0, -2),
        targets.flatten(),
        reduction="none",
        ignore_index=pad_token_id,
    ).view_as(targets)
