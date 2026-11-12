import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, TrainerCallback, set_seed

from model.data_utils import SCFTDataset, SCFTCollator
from model.losses import user_pos_neg_infonce_loss
from model.model_utils import load_model_and_tokenizer
from model.prompts import read_jsonl


class SCFTTrainer(Trainer):
    def __init__(self, temperature: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def save_model(self, output_dir, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

    def _encode(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        masked = hidden_states * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = masked.sum(dim=1) / lengths
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        user_input_ids = inputs["user_input_ids"]
        user_attention_mask = inputs["user_attention_mask"]
        pos_input_ids = inputs["pos_input_ids"]
        pos_attention_mask = inputs["pos_attention_mask"]
        neg_input_ids = inputs["neg_input_ids"]
        neg_attention_mask = inputs["neg_attention_mask"]
        user_embeddings = self._encode(
            model=model,
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
        )
        pos_embeddings = self._encode(
            model=model,
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
        )
        neg_embeddings = self._encode(
            model=model,
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
        )
        loss = user_pos_neg_infonce_loss(
            user_embeddings=user_embeddings,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            temperature=self.temperature,
        )
        if return_outputs:
            return loss, {
                "user_embeddings": user_embeddings,
                "pos_embeddings": pos_embeddings,
                "neg_embeddings": neg_embeddings,
            }
        return loss


class TimeLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.steps = []
        self.losses = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if state.max_steps:
            ratio = state.global_step / max(state.max_steps, 1)
            total = elapsed / max(ratio, 1e-8)
            remaining = total - elapsed
            print(
                f"[time] elapsed={elapsed/3600:.2f}h "
                f"total_est={total/3600:.2f}h "
                f"remaining={remaining/3600:.2f}h "
                f"steps={state.global_step}/{state.max_steps}"
            )
        else:
            print(
                f"[time] elapsed={elapsed/3600:.2f}h "
                f"steps={state.global_step}"
            )
        if logs is not None and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(float(logs["loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.steps:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        steps = self.steps
        losses = self.losses
        alpha = 0.1
        smoothed = []
        for i, value in enumerate(losses):
            if i == 0:
                smoothed.append(value)
            else:
                smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(steps, losses, label="original", alpha=0.3)
        ax.plot(steps, smoothed, label="smoothed")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"training loss of {args.output_dir}")
        ax.legend()
        output_dir = args.output_dir or "."
        path = os.path.join(output_dir, "training_loss.png")
        fig.savefig(path)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--use_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_dataset = read_jsonl(Path(args.train_file))
    print("num raw examples:", len(raw_dataset))
    lora_target_modules = [
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    ]
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
    )
    train_dataset = SCFTDataset(
        rows=raw_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    if not len(train_dataset):
        raise ValueError("SCFT data must contain users with both positive and negative items")
    print("num triplets:", len(train_dataset))
    data_collator = SCFTCollator(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        bf16=args.bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to=[],
    )
    trainer = SCFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        temperature=args.temperature,
        callbacks=[TimeLoggingCallback()],
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
