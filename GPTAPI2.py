# 4o-mini | max context window: 128,000 tokens | max output tokens: 16,384
from openai import OpenAI
import random
from dataloader import get_training_data, get_test_data
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Prompts options
prompt_options = {
    "explain": """Task: Analyze two responses to an original post that try to change the original poster's view.
    Focus on comparing stylistic elements since they have high Jaccard similarity (content overlap). Consider:
   - Tone and emotional approach
   - Rhetorical techniques
   - Sentence structure and readability
   - Engagement with the original poster""",
    "predict": """Now, based on your analysis, predict which response successfully changed the original poster's view.
    1. Provide your prediction by stating either "$$Answer: First Response$$" or "$$Answer: Second Response$$
    2. Briefly explain the key factor(s) that led to your prediction.""",
}

# Get data
# ------------------------------------------------------------------------------
prompts, labels = get_test_data()

def get_sample(prompts, labels, n_samples=500):
    random.seed(42)
    indices = list(range(len(prompts)))
    sampled_indices = random.sample(indices, n_samples)
    sampled_prompts = [prompts[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    return sampled_prompts, sampled_labels

# Get same 500 samples
sampled_prompts, sampled_labels = get_sample(prompts, labels, n_samples=500)

client = OpenAI(
    api_key=api_key
)

# Run classification
# ------------------------------------------------------------------------------
correct = 0
total = 0

output_file = "explain_then_predict_separate.jsonl"
system_prompt = prompt_options["explain"]

for i in range(len(sampled_prompts)):
    prompt = sampled_prompts[i]
    true_label = sampled_labels[i]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        response = completion.choices[0].message.content.strip()
        print(response)

        completion_2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
                {"role": "user", "content": prompt_options["predict"]}
            ],
        )
        response_2 = completion_2.choices[0].message.content.strip()
        print(response_2)

        if "$$answer: first response$$" in response_2.lower():
            pred_label = "a"
        elif "$$answer: second response$$" in response_2.lower():
            pred_label = "b"
        else:
            pred_label = None

        result = {
            "index": i,
            "prompt": prompt,
            "true_label": true_label,
            "predicted_label": pred_label,
            "explanation": response,
            "prediction": response_2,
            "is_correct": pred_label == true_label if pred_label is not None else None,
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        if pred_label is not None:
            total += 1
            if pred_label == true_label:
                correct += 1

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1} examples")

    except Exception as e:
        print(f"An error occurred at index {i}: {e}")
        error_result = {
            "index": i,
            "prompt": prompt,
            "true_label": true_label,
            "error": str(e),
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(error_result) + "\n")
        continue

# Results
# ------------------------------------------------------------------------------
if total > 0:
    print(f"Valid predictions: {total}")
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
else:
    print("No valid predictions were made.")

"""
Zero shot basic prompt baseline:
--------------------------------------------------------------------------------

Results:
--------------------------------------------------------------------------------
Valid predictions is 493/500, and accuracy is 58.82%

Prompt:
--------------------------------------------------------------------------------
Task: Analyze two responses to an original post and predict which successfully changed the original poster's view.

1. Compare the responses, focusing on stylistic elements since they have high Jaccard similarity (content overlap). Consider:
   - Tone and emotional approach
   - Rhetorical techniques
   - Sentence structure and readability
   - Engagement with the original poster

2. Provide your prediction by stating either "$$Answer: First Response$$" or "$$Answer: Second Response$$"

3. Briefly explain the key factor(s) that led to your prediction.

Original Post Title: 
{entry['op_title']}

Original Post: 
{entry['op_text']}

First Response: 
{entry['argument_a']}

Second Response: 
{entry['argument_b']}

Results:
--------------------------------------------------------------------------------
Valid predictions is 500/500, and accuracy is 57.80%
"""
