import gc
import random

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import wandb

from minrl.grpo import process_generations, compute_loss, Episode
from minrl.model import LanguageModel, generate
from minrl.tasks.countdown import CountdownTask
from minrl.utils import TrainingTimer


class Trainer:
    def __init__(
        self,
        model: LanguageModel,
        task: CountdownTask,
        lr: float = 1e-5,
        beta: float = 0.0,
        max_grad_norm: float | None = None,
        num_answers_per_question: int = 16,
        generation_length: int = 512,
        gradient_batch_size: int = 2,
        max_steps: int = 1000,
        eval_interval: int = 25,
        ignore_incomplete_responses: bool = False,
    ):
        # Training parameters
        self.lr = lr
        self.beta = beta  # regularization parameter
        self.task = task
        self.generation_length = generation_length
        self.max_grad_norm = max_grad_norm
        self.num_answers_per_question = num_answers_per_question
        self.max_steps = max_steps
        self.gradient_batch_size = gradient_batch_size
        self.ignore_incomplete_responses = ignore_incomplete_responses
        self.eval_interval = eval_interval

        # Training state
        self.model = model
        self.current_step = 0
        self.optimizer = None
        self.training_timer = TrainingTimer()

    def run_training(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} of {total_params:,} total parameters")

        data_loader = iter(self.task.train_data_loader())

        if self.optimizer is None:
            self.optimizer = self.configure_optimizer()

        while self.current_step < self.max_steps:
            if self.current_step % self.eval_interval == 0:
                with self.training_timer("eval"):
                    self.run_evaluation()

            batch = next(data_loader)
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}

            with self.training_timer("rollout"):
                episodes = self.rollout(
                    batch, ignore_incomplete_responses=self.ignore_incomplete_responses
                )

            if len(episodes) == 0:
                raise ValueError("No episodes generated")

            with self.training_timer("backprop"):
                total_loss = self.backprop(batch, episodes)

            with self.training_timer("optimizer_step"):
                self.optimizer_step()

            self.log(episodes, total_loss, batch)
            self.training_timer.print_timings()
            self.current_step += 1

    @torch.no_grad()
    def run_evaluation(self):
        is_training = self.model.training
        self.model.eval()

        print("Running evaluation...")
        data_loader = iter(self.task.test_data_loader())

        all_episodes = []
        for batch in tqdm(data_loader):
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            episodes = self.rollout(
                batch, nrepeats=1, normalize=False, ignore_incomplete_responses=False
            )
            all_episodes += episodes

        all_episodes = sorted(all_episodes, key=lambda ep: ep.reward)
        avg_reward = np.mean([ep.reward for ep in all_episodes])
        pct_over_0_8 = np.mean([ep.reward > 0.8 for ep in all_episodes])
        print(f"Avg reward: {avg_reward:.3f}")
        print(f"Percent over 0.8: {pct_over_0_8:.3f}")
        wandb.log(
            {"eval/avg_reward": avg_reward, "eval/pct_over_0_8": pct_over_0_8},
            step=self.current_step,
        )

        self.model.train(is_training)

    def rollout(
        self,
        batch: dict[str, Tensor],
        nrepeats=None,
        normalize=True,
        ignore_incomplete_responses=True,
    ) -> list[Episode]:
        if nrepeats is None:
            nrepeats = self.num_answers_per_question

        repeated_prompt_ids = batch["prefix_token_ids"]
        if nrepeats > 1:
            repeated_prompt_ids = repeated_prompt_ids.repeat_interleave(nrepeats, dim=0)

        with torch.no_grad():
            token_ids = generate(
                self.model,
                repeated_prompt_ids,
                generation_length=self.generation_length,
                pad_token_id=self.task.tokenizer.pad_token_id,
                eos_token_id=self.task.tokenizer.eos_token_id,
                early_stopping=False,
            )
            outputs = {"prompt_ids": repeated_prompt_ids, "token_ids": token_ids}
            episodes = process_generations(
                task=self.task,
                batch=batch,
                outputs=outputs,
                normalize=normalize,
                ignore_incomplete_responses=ignore_incomplete_responses,
            )
            random.shuffle(episodes)

        return episodes

    def backprop(self, batch: dict[str, Tensor], episodes: list[Episode]):
        epoch_token_ids = torch.stack([ep.token_ids for ep in episodes]).to("cuda")
        epoch_generated_ids = torch.stack([ep.generated_ids for ep in episodes]).to("cuda")
        epoch_advantages = torch.tensor([ep.advantage for ep in episodes], device="cuda")

        dataset = {
            "token_ids": epoch_token_ids,
            "generated_ids": epoch_generated_ids,
            "advantages": epoch_advantages,
        }

        total_loss = 0.0
        total_ntokens = 0
        normalize_factor = 1024  # make sure gradients don't explode as the loss is sum over tokens
        for i in range(0, len(episodes), self.gradient_batch_size):
            batch = {k: v[i : i + self.gradient_batch_size] for k, v in dataset.items()}
            loss, ntokens = compute_loss(
                model=self.model,
                token_ids=batch["token_ids"],
                generated_ids=batch["generated_ids"],
                advantages=batch["advantages"],
                pad_token_id=self.task.tokenizer.pad_token_id,
                beta=self.beta,
            )
            loss = loss / normalize_factor
            loss.backward()
            total_loss = total_loss + loss.detach()
            total_ntokens = total_ntokens + ntokens

        # Normalize gradients to number of tokens
        grad_scale = total_ntokens / normalize_factor
        for pn, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad /= grad_scale

        return total_loss / grad_scale

    def optimizer_step(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def log(self, episodes: list[Episode], total_loss: Tensor, batch: dict[str, Tensor]):
        avg_reward = np.mean([ep.reward for ep in episodes])
        min_reward = min([ep.reward for ep in episodes])
        max_reward = max([ep.reward for ep in episodes])

        if self.current_step % 20 == 0:
            print()
            print("======")
            best_episode = max(episodes, key=lambda ep: ep.reward)
            print("Best episode:")
            print(best_episode.response)
            worst_episode = min(episodes, key=lambda ep: ep.reward)
            print("Worst episode:")
            print(worst_episode.response)
            print("======")

        print(
            f"[{self.current_step:05d}] reward avg={avg_reward:.3f} min={min_reward:.3f} "
            f"max={max_reward:.3f} n={len(episodes)}"
        )
        print(f"[{self.current_step:05d}] loss avg={total_loss.item():.5f}")

        log_data = {
            "batch_idx": self.current_step,
            "reward/avg": avg_reward,
            "reward/median": np.median([ep.reward for ep in episodes]),
            "reward/min": min_reward,
            "reward/max": max_reward,
            "episodes/num": len(episodes),
            "generated_tokens/avg": np.mean(
                [ep.num_generated_tokens for ep in episodes if ep.num_generated_tokens]
            ),
        }

        if len(episodes) > 3:
            sorted_episodes = sorted(episodes, key=lambda ep: ep.reward)
            log_data["sample_episodes"] = create_wandb_table(
                [
                    sorted_episodes[0],
                    sorted_episodes[len(sorted_episodes) // 2],
                    sorted_episodes[-1],
                ],
                step=self.current_step,
                task=self.task,
            )

        wandb.log(log_data, step=self.current_step)

    def configure_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=0.01
        )


def create_wandb_table(episodes: list[Episode], task, step: int) -> wandb.Table:
    table = wandb.Table(
        columns=["step", "prompt", "response", "reward", "advantage", "generated_tokens"]
    )

    for episode in episodes:
        prompt = task.tokenizer.decode(episode.prompt_ids, True)
        response = task.tokenizer.decode(episode.generated_ids, True)
        table.add_data(
            step,
            prompt,
            response,
            episode.reward,
            episode.advantage,
            episode.num_generated_tokens,
        )

    return table
