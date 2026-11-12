import json
import pickle


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_prompt_inputs(directory):
    with (directory / "train.pkl").open("rb") as file:
        train = pickle.load(file)
    users = read_jsonl(directory / "users.jsonl")
    items = read_jsonl(directory / "items.jsonl")
    if set(train) != set(range(len(users))):
        raise ValueError("train.pkl and users.jsonl must use the same contiguous user IDs")
    if any(not 0 <= i < len(items) for history in train.values() for i in history):
        raise ValueError("Training item ID missing from items.jsonl")
    return train, users, items


def item_prompt(item):
    text = f"item {item['item']} is {item['item_name']}"
    if "item_brand" in item:
        text += f" and the brand is {item['item_brand']}"
    return text + f" and the item categorie is {item['item_category']}"


def history_prompt(user, history, items):
    fields = [
        ("buys", "item_name"),
        ("whose id are", "item"),
        ("and the item brands are", "item_brand"),
        ("and the item categories are", "item_category"),
    ]
    return " ".join(
        [f"user {user['user']}"] + [
            phrase + " " + " ".join(items[i].get(key, "") for i in history)
            for phrase, key in fields
        ]
    )
