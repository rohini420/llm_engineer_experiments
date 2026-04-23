text_sequences = ["We had a picnic on the river bank.", "Tom deposited his paycheck at the savings bank account."]

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from transformers import BertTokenizer

def ensure_nltk_data(resources):
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

ensure_nltk_data({
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
})

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Step 2: BERT tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    print("Original Text:", text)
    tokens = word_tokenize(text)
    print("Tokens:", tokens)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    print("Tokens we have after removing stop words, punctuation and lemmatization:\n", tokens, "\n")
    return ' '.join(tokens)

preprocessed_text_sequences = [preprocess_text(text_sequence) for text_sequence in text_sequences]
print("Preprocessed Text Sequences:", preprocessed_text_sequences)

# Tokenize the preprocessed text sequences using BERT tokenizer
inputs = tokenizer(preprocessed_text_sequences, return_tensors='pt', padding=True, truncation=True)
print("BERT Tokenized Inputs:", inputs)

