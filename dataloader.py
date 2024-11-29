import json
import bz2
import random

# load data
# ------------------------------------------------------------------------------
def process_cmv_pairs(file_path: str) -> list[dict[str, str]]:
    """Process the CMV pair data from the bz2 file. Randomize ordering of the
    delta vs non-delta counterargument."""
    random.seed(42)
    processed_data = []

    with bz2.open(file_path, "rt", encoding="utf-8") as file:
        for line in file:
            pair = json.loads(line.strip())
            successful_argument = pair["positive"]["comments"][0]["body"]
            unsuccessful_argument = pair["negative"]["comments"][0]["body"]

            if random.random() < 0.5:
                argument_a = successful_argument
                argument_b = unsuccessful_argument
                label = "a"
            else:
                argument_a = unsuccessful_argument
                argument_b = successful_argument
                label = "b"

            entry = {
                "op_title": pair["op_title"],
                "op_text": pair["op_text"],
                "argument_a": argument_a,
                "argument_b": argument_b,
                "label": label,
            }
            processed_data.append(entry)

    return processed_data

# Format training examples into prompts
# ------------------------------------------------------------------------------
def prompt(entry: dict[str, str]) -> str:
    """
    Format training example into LLM prompt
    """
    prompt = f"""

Original Post Title: 
{entry['op_title']}

Original Post: 
{entry['op_text']}

First Response: 
{entry['argument_a']}

Second Response: 
{entry['argument_b']}

"""
    return prompt

# Get model prompts and labels
# ------------------------------------------------------------------------------
def get_training_data(file_path: str = "pair_task/train_pair_data.jsonlist.bz2") -> tuple[list[str], list[str]]:
    """Get processed prompts and labels ready for model input."""
    processed_data = process_cmv_pairs(file_path)
    model_prompts = [prompt(entry) for entry in processed_data]
    model_labels = [entry["label"] for entry in processed_data]
    return model_prompts, model_labels

def get_test_data(file_path: str = "pair_task/heldout_pair_data.jsonlist.bz2") -> tuple[list[str], list[str]]:
    """Get processed prompts and labels ready for model input."""
    processed_data = process_cmv_pairs(file_path)
    model_prompts = [prompt(entry) for entry in processed_data]
    model_labels = [entry["label"] for entry in processed_data]
    return model_prompts, model_labels
