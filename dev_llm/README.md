# `dev_llm`

Small LLM-focused experiments around prompting, local inference, Hugging Face pipelines, and lightweight evaluation.

## Projects

- `analyzing_car_reviews_with_llms.py`
  Multi-task Hugging Face workflow for sentiment analysis, translation, question answering, summarization, and metric-based evaluation.
- `classifying_emails_using_llama.py`
  Prompt-based email routing with a local GGUF model loaded through `llama_cpp`.
- `service_desk_ticket_classification_with_deep_learning.py`
  A compact PyTorch text classifier for multiclass ticket prediction.

## Libraries used

- `transformers`
- `evaluate`
- `llama-cpp-python`
- `torch`, `torchmetrics`
- `pandas`, `numpy`, `scikit-learn`

## What I practiced here

- turning natural-language tasks into prompt templates
- comparing pretrained-model workflows with custom neural models
- evaluating outputs with `accuracy`, `f1`, `precision`, and `recall`
- working with local model files and course-provided datasets

## Running notes

- These scripts expect course datasets or model files that are not stored in this repo.
- `classifying_emails_using_llama.py` also expects a local `.gguf` model path.
