"""Multi-step Hugging Face workflow for review sentiment, translation, question answering, and summarization."""

# Import necessary packages
import pandas as pd
import torch
import evaluate
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

from transformers import logging
logging.set_verbosity(logging.WARNING)

file_path = "data/car_reviews.csv"
df = pd.read_csv(file_path, delimiter=";")
reviews = df['Review'].tolist()
real_labels = df['Class'].tolist()

# Start your code here!
from transformers import pipeline
model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
predicted_labels = model(reviews)
references = [1 if label == "POSITIVE" else 0 for label in real_labels]
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]
accuracy_result_dict = accuracy.compute(references=references, predictions=predictions)
accuracy_result = accuracy_result_dict['accuracy']
f1_result_dict = f1.compute(references=references, predictions=predictions)
f1_result = f1_result_dict['f1']
first_review = reviews[0]
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translated_review = translator(first_review, max_length=27)[0]['translation_text']
with open("data/reference_translations.txt", 'r') as file:
    lines = file.readlines()
references = [line.strip() for line in lines]
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=[translated_review], references=[references])
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering


model_ckp = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckp)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckp)

context = reviews[1]
question = "What did he like about the brand?"
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print("Answer: ", answer)


text_to_summarize = reviews[-1]
model_name = "cnicu/t5-small-booksum"
summarizer = pipeline("summarization", model=model_name)
outputs = summarizer(text_to_summarize, max_length=53)
summarized_text = outputs[0]['summary_text']
print(f"Summarized text:\n{summarized_text}")
