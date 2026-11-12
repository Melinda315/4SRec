"""Load a ServiceLLM adapter and extract an ordered user-then-item tensor."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm

from model.model_utils import load_base_and_tokenizer
from model.prompts import history_prompt, item_prompt, load_prompt_inputs


@torch.inference_mode()
def encode(model, tokenizer, texts, batch_size, max_length):
    rows = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embed"):
        batch = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(model.get_input_embeddings().weight.device)
        outputs = model(
            **batch, output_hidden_states=True, use_cache=False, logits_to_keep=1, return_dict=True
        )
        hidden = outputs.hidden_states[-1].float()
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.device)
        rows.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).cpu())
    return torch.cat(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.batch_size, args.max_length) < 1:
        parser.error("batch-size and max-length must be positive")
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("--adapter must point to an adapter saved by train_scft_lora.py")

    train, users, items = load_prompt_inputs(args.input_dir)
    texts = [history_prompt(user, train[i], items) for i, user in enumerate(users)]
    texts.extend(item_prompt(item) for item in items)
    base, tokenizer = load_base_and_tokenizer(args.base_model, args.use_4bit, args.bf16)
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    tensor = encode(model, tokenizer, texts, args.batch_size, args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, args.output)
    print(f"raw_embedding_shape={tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
