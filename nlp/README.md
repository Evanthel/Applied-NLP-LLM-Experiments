# `nlp`

Classical NLP experiments covering text preprocessing, feature engineering, sentiment analysis, speech-to-text, and named entity extraction.

## Projects

- `sentiment_analysis_in_python.py`
  Word clouds, tokenization, TF-IDF features, n-grams, and logistic regression for review sentiment.
- `support_calls.py`
  Audio inspection, Google Speech Recognition transcription, VADER sentiment scoring, spaCy NER, and similarity search on support-call transcripts.

## Libraries used

- `nltk`
- `scikit-learn`
- `spacy`
- `SpeechRecognition`
- `pydub`
- `matplotlib`, `wordcloud`
- `pandas`

## What I practiced here

- building classical NLP baselines before reaching for LLMs
- feature extraction with TF-IDF and n-grams
- sentiment scoring and entity extraction on noisy text
- combining audio preprocessing with downstream text analysis

## Running notes

- Install dependencies from the repo root with `pip install -r requirements.txt`.
- `support_calls.py` also needs the spaCy English model:
  `python -m spacy download en_core_web_sm`
