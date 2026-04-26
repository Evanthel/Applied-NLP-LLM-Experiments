# `nlp`

Classical NLP experiments covering text preprocessing, feature engineering, sentiment analysis, speech-to-text, named entity extraction, and notebook-based toolkit practice with NLTK and spaCy.

## Projects

- `sentiment_analysis_in_python.py`
  Word clouds, tokenization, TF-IDF features, n-grams, and logistic regression for review sentiment.
- `support_calls.py`
  Audio inspection, Google Speech Recognition transcription, VADER sentiment scoring, spaCy NER, and similarity search on support-call transcripts.
- `nltk.ipynb`
  Combined NLTK notebook covering tokenization, stopword removal, stemming, lemmatization, POS tagging, NER, and Shakespeare corpus analysis.
- `spacy.ipynb`
  Combined spaCy notebook covering tokenization, stopword filtering, lemmatization, POS tagging, sentence similarity, and Shakespeare corpus analysis.
- `question_answering_with_web_search.ipynb`
  Small retrieval-style QA exercise using spaCy, `requests`, BeautifulSoup, and Wikipedia pages.

## Libraries used

- `nltk`
- `scikit-learn`
- `spacy`
- `requests`
- `beautifulsoup4`
- `SpeechRecognition`
- `pydub`
- `matplotlib`, `wordcloud`
- `pandas`

## What I practiced here

- building classical NLP baselines before reaching for LLMs
- feature extraction with TF-IDF and n-grams
- sentiment scoring and entity extraction on noisy text
- comparing NLTK and spaCy on similar introductory workflows
- lightweight retrieval and answer extraction without relying on an LLM to answer directly
- combining audio preprocessing with downstream text analysis

## Running notes

- Install dependencies from the repo root with `pip install -r requirements.txt`.
- `support_calls.py` also needs the spaCy English model:
  `python -m spacy download en_core_web_sm`
- `nltk.ipynb` and `spacy.ipynb` expect a local `pg100.txt` file from Project Gutenberg.
- `question_answering_with_web_search.ipynb` needs internet access for Wikipedia requests.
