# Import required libraries
import os
import pandas as pd
from llama_cpp import Llama


# Load the email dataset
emails_df = pd.read_csv("data/email_categories_data.csv")

# Display the first few rows of our dataset
print("Preview of our email dataset:")
print(emails_df.head(2).to_string(index=False))

# Set the local model path via environment variable before running the script.
model_path = os.environ.get("LLAMA_MODEL_PATH")
if not model_path:
    raise ValueError("Set LLAMA_MODEL_PATH to the local GGUF model path before running this script.")

llm = Llama(model_path=model_path)
prompt = """
Email routing categories:

1. Priority - Important or time-sensitive messages that require user attention or action.
2. Updates - Notifications or informational messages related to existing services, accounts, or subscriptions.
3. Promotions - Marketing, advertisements, or sales-related messages.

Use the subject and body text to choose the most appropriate category.
Return one label only: Priority, Updates, or Promotions.
"""


def process_message(llm, message, prompt):
    full_prompt = f"""{prompt}

Email:
{message}

Category:"""
    response = llm(full_prompt, max_tokens=16, temperature=0, stop=["\n"])
    return response["choices"][0]["text"].strip()


test_emails = emails_df.head(2)
results = []
for idx, row in test_emails.iterrows():
    email_content = row["email_content"]
    expected_category = row["expected_category"]

    # Get model's classification
    result = process_message(llm, email_content, prompt)

    # Store results
    results.append({
        "email_content": email_content,
        "expected_category": expected_category,
        "model_output": result,
    })
results_df = pd.DataFrame(results)

result1 = results_df["model_output"].iloc[0]
result2 = results_df["model_output"].iloc[1]

print(f"Result 1: `{result1}`\nResult 2: `{result2}`")
