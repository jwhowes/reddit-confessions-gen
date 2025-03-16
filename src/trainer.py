import os
import yaml
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from .config import BaseConfig
from .model.transformer import TransformerConfig, Transformer
from .data import RedditConfessionsDatasetConfig, get_dataloader


accelerator = Accelerator()


@dataclass
class TrainerConfig(BaseConfig):
    log_interval: int = 100

    lr: float = 5e-5
    weight_decay: float = 0.01
    clip_grad: float = 3.0


class Trainer:
    @accelerator.main_process_first()
    def __init__(self, config_dir: str):
        exp_name = os.path.basename(config_dir)

        self.exp_dir = os.path.join("experiments", exp_name)
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.checkpoints_dir = os.path.join(self.exp_dir, "checkpoints")
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.log_path = os.path.join(self.exp_dir, "log.csv")
        with open(self.log_path, "w+") as f:
            f.write("ckpt,average loss,timestamp\n")

        self.model_config = TransformerConfig.from_yaml(os.path.join(config_dir, "model.yaml"))
        with open(os.path.join(self.exp_dir, "model.yaml"), "w+") as f:
            yaml.dump(self.model_config, f)

        self.data_config = RedditConfessionsDatasetConfig.from_yaml(os.path.join(config_dir, "data.yaml"))
        with open(os.path.join(self.exp_dir, "data.yaml"), "w+") as f:
            yaml.dump(self.data_config, f)

        self.train_config = TrainerConfig.from_yaml(os.path.join(config_dir, "train.yaml"))
        with open(os.path.join(self.exp_dir, "train.yaml"), "w+") as f:
            yaml.dump(self.train_config, f)

        self.ckpt = 1

    def log(self, model: Transformer, average_loss: float):
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            torch.save(
                accelerator.get_state_dict(model),
                os.path.join(self.checkpoints_dir, f"checkpoint_{self.ckpt:03}.pt")
            )

            with open(self.log_path, "a") as f:
                f.write(f"{self.ckpt},{average_loss},{datetime.now()}\n")

        self.ckpt += 1

    def train(self):
        model = Transformer.from_config(self.model_config)

        dataloader = get_dataloader(self.data_config, self.model_config.tokenizer_path, self.model_config.max_length)

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.train_config.lr, weight_decay=self.train_config.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=len(dataloader)
        )

        model, dataloader, opt, lr_scheduler = accelerator.prepare(
            model, dataloader, opt, lr_scheduler
        )

        total_loss = 0
        prev_ckpt = 0
        model.train()
        for i, encoding in enumerate(dataloader):
            opt.zero_grad()

            pred = model(encoding["prefix"])
            loss = F.cross_entropy(pred.transpose(1, 2), encoding["suffix"], reduction="none")
            loss[encoding["pad_mask"]] = 0
            loss = loss.sum() / (~encoding["pad_mask"]).sum()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), self.train_config.clip_grad)

            opt.step()
            lr_scheduler.step()

            total_loss += loss.item()

            if i % self.train_config.log_interval == 0:
                average_loss = total_loss / (i - prev_ckpt + 1)

                if accelerator.is_main_process:
                    print(f"{i} / {len(dataloader)} iters.\tAverage loss: {average_loss:.4f}")

                self.log(model, average_loss)

                total_loss = 0
                prev_ckpt = i + 1
