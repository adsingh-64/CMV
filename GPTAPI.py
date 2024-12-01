import openai
import random
from dataloader import get_training_data, get_test_data
import json
import os
from dotenv import load_dotenv

# Set-up
# ------------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompts
# ------------------------------------------------------------------------------
prompt_options = {
    "basic": "Task: Analyze two responses to an original post and predict which successfully changed the original poster's view. Provide your prediction by stating either '$$Answer: First Response$$' or '$$Answer: Second Response$$'",
    "explain_then_predict": """Task: Analyze two responses to an original post and predict which successfully changed the original poster's view.

    1. Compare the responses, focusing on stylistic elements since they have high Jaccard similarity (content overlap). Consider:
   - Tone and emotional approach
   - Rhetorical techniques
   - Sentence structure and readability
   - Engagement with the original poster

    2. Provide your prediction by stating either "$$Answer: First Response$$" or "$$Answer: Second Response$$"

    3. Briefly explain the key factor(s) that led to your prediction.""",
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

# Run classification
# ------------------------------------------------------------------------------
correct = 0
total = 0

output_file = "explain_then_predict_together.jsonl"
system_prompt = prompt_options["basic"]

with open(output_file, "a") as f:
    for i in range(len(sampled_prompts)):
        prompt = sampled_prompts[i]
        true_label = sampled_labels[i]

        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            response = completion.choices[0].message.content.strip()

            if "$$answer: first response$$" in response.lower(): # should use regex pattern
                pred_label = "a"
            elif "$$answer: second response$$" in response.lower():
                pred_label = "b"
            else:
                pred_label = None

            result = {
                "index": i,
                "prompt": prompt,
                "true_label": true_label,
                "predicted_label": pred_label,
                "full_response": response,
                "is_correct": pred_label == true_label if pred_label is not None else None,
            }

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