# Applied NLP & LLM Experiments

Hands-on Python experiments built while working through DataCamp projects and NLP/LLM exercises. The repo is less about polished apps and more about learning by implementing: prompting, classical NLP, local LLM inference, evaluation, and small deep learning prototypes.

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Notes:

- Some scripts still require course datasets or local model files that are not committed to the repo.
- `dev_llm/classifying_emails_using_llama.py` expects a local `.gguf` model path.
- FashionMNIST downloads automatically when you run the image-classification project.

## Data / Models expected

- `dev_llm/analyzing_car_reviews_with_llms.py`
  Expects `data/car_reviews.csv` and `data/reference_translations.txt`.
- `dev_llm/classifying_emails_using_llama.py`
  Expects `data/email_categories_data.csv` and a local GGUF model file referenced by `model_path`.
- `dev_llm/service_desk_ticket_classification_with_deep_learning.py`
  Expects `words.json`, `text.json`, and `labels.npy`.
- `nlp/support_calls.py`
  Expects `sample_customer_call.wav` and `customer_call_transcriptions.csv`.
- `nlp/sentiment_analysis_in_python.py`
  Assumes an in-memory `reviews` DataFrame and `positive_reviews` text prepared by the course exercise.
- `calorie_intake_calculator/calorie_intake_calculator.py`
  Expects `nutrition.json`.
- `ecommerce_clothing_classifier/ecommerce_clothing_classifier.py`
  Downloads FashionMNIST automatically; no manual dataset setup is needed.
- `interstellar_delivery_datetime/interstellar_delivery_datetime.py`
  No external data required.

## What this repo shows

- Prompt-based classification with local LLMs via `llama.cpp`
- Hugging Face pipelines for sentiment analysis, translation, summarization, and question answering
- Classical NLP workflows with TF-IDF, n-grams, logistic regression, VADER, and spaCy
- PyTorch fundamentals through custom text and image classifiers
- Basic Python problem solving beyond NLP, including JSON data processing and `datetime` utilities

## What I learned

- How to turn a natural language task into a concrete prompting or classification setup
- When to use a lightweight classical NLP baseline versus an LLM-powered approach
- How to evaluate model behavior with `accuracy`, `f1`, `precision`, `recall`, and `BLEU`
- How to preprocess text data with tokenization, padding, vectorization, and simple feature engineering
- How to work with local models, pretrained transformers, and small custom neural networks in one workflow

## Tech and libraries used

- Python
- pandas, numpy
- scikit-learn
- PyTorch, torchmetrics, torchvision
- Hugging Face `transformers`
- `evaluate`
- `llama-cpp-python`
- NLTK, spaCy
- SpeechRecognition, pydub

## Project highlights

### [`dev_llm/`](./dev_llm)

This folder is the strongest part of the repo from an LLM experimentation perspective.

- [`analyzing_car_reviews_with_llms.py`](./dev_llm/analyzing_car_reviews_with_llms.py)
  Uses pretrained Hugging Face models for sentiment classification, translation, extractive question answering, and summarization. It also evaluates outputs with metrics instead of stopping at a demo.
- [`classifying_emails_using_llama.py`](./dev_llm/classifying_emails_using_llama.py)
  Runs local inference with a GGUF model through `llama_cpp` and uses a prompt-driven category classifier for email routing.
- [`service_desk_ticket_classification_with_deep_learning.py`](./dev_llm/service_desk_ticket_classification_with_deep_learning.py)
  Builds a small PyTorch text classifier with embeddings and 1D convolution, then evaluates multiclass performance.
- Folder guide: [`dev_llm/README.md`](./dev_llm/README.md)

### [`nlp/`](./nlp)

- [`sentiment_analysis_in_python.py`](./nlp/sentiment_analysis_in_python.py)
  Covers word clouds, tokenization, TF-IDF, n-grams, and logistic regression for sentiment analysis.
- [`support_calls.py`](./nlp/support_calls.py)
  Combines speech-to-text, VADER sentiment analysis, named entity recognition, and semantic similarity on support call data.
- Folder guide: [`nlp/README.md`](./nlp/README.md)

### Other small builds

- [`calorie_intake_calculator/`](./calorie_intake_calculator)
  A small JSON-driven nutrition calculator that practices data access and reusable function design.
- [`ecommerce_clothing_classifier/`](./ecommerce_clothing_classifier)
  A compact CNN image classifier trained on FashionMNIST with PyTorch.
- Folder guide: [`ecommerce_clothing_classifier/README.md`](./ecommerce_clothing_classifier/README.md)
- [`interstellar_delivery_datetime/`](./interstellar_delivery_datetime)
  Small utility functions around timestamps, durations, and delivery-date calculations.

## Why this repo is interesting

It mixes:

- classical NLP and modern LLM workflows
- local model inference and pretrained tooling
- experimentation and evaluation
- text, audio, and image tasks

That combination makes it a good learning log for practical model intuition, not just a dump of isolated exercises.

## Notes

- Most files are short project scripts exported from course exercises rather than full production-ready packages.
- Some scripts expect course datasets, downloaded model weights, or local paths that are not included in the repository.
- Folder and file names are standardized to lowercase `snake_case` to keep paths predictable and link-friendly.
- The repo now includes `requirements.txt` and an MIT `LICENSE` to make reuse and setup clearer.
