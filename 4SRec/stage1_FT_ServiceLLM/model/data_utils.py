from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SCFTDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets: List[Dict[str, Any]] = []

        user_map: Dict[str | int, Dict[str, Any]] = {}
        for row in rows:
            user_id = row["user_id"]
            user_text = row["user_text"]
            item_text = row["item"]
            label = row.get("label", "pos")
            entry = user_map.get(user_id)
            if entry is None:
                entry = {
                    "user_text": user_text,
                    "pos_items": [],
                    "neg_items": [],
                }
                user_map[user_id] = entry
            if label == "pos":
                entry["pos_items"].append(item_text)
            else:
                entry["neg_items"].append(item_text)

        for entry in user_map.values():
            if entry["pos_items"] and entry["neg_items"]:
                pos_items = entry["pos_items"]
                neg_items = entry["neg_items"]
                for pos_item in pos_items:
                    for neg_item in neg_items:
                        self.triplets.append(
                            {
                                "user_text": entry["user_text"],
                                "pos_item": pos_item,
                                "neg_item": neg_item,
                            }
                        )

        self.num_triplets = len(self.triplets)

    def __len__(self) -> int:
        return self.num_triplets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        triplet = self.triplets[idx]
        result = {}
        for side, key in (("user", "user_text"), ("pos", "pos_item"), ("neg", "neg_item")):
            encoded = self.tokenizer(
                triplet[key], max_length=self.max_length, truncation=True,
                padding=False, return_tensors="pt",
            )
            for field in ("input_ids", "attention_mask"):
                result[f"{side}_{field}"] = encoded[field].squeeze(0)
        return result


class SCFTCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def pad_side(key_ids: str, key_mask: str) -> Dict[str, torch.Tensor]:
            inputs = {
                "input_ids": [f[key_ids] for f in features],
                "attention_mask": [f[key_mask] for f in features],
            }
            batch = self.tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            )
            return batch

        user_batch = pad_side("user_input_ids", "user_attention_mask")
        pos_batch = pad_side("pos_input_ids", "pos_attention_mask")
        neg_batch = pad_side("neg_input_ids", "neg_attention_mask")

        return {
            "user_input_ids": user_batch["input_ids"],
            "user_attention_mask": user_batch["attention_mask"],
            "pos_input_ids": pos_batch["input_ids"],
            "pos_attention_mask": pos_batch["attention_mask"],
            "neg_input_ids": neg_batch["input_ids"],
            "neg_attention_mask": neg_batch["attention_mask"],
        }


