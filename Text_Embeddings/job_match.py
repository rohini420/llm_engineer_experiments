import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Load dataset from CSV file
df = pd.read_csv("/Users/rbhatlapenumarthi/Downloads/job_title_des.csv")
# Read only the first twenty job descriptions
df = df.head(20)
# Preprocess text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    return ' '.join(tokens)

df['processed_description'] = df['Job Description'].apply(preprocess)

# Load pretrained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embedding for a text using [CLS] token
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
    return cls_embedding.numpy()

# Tokenize text

# Feed into BERT

# Extract [CLS] embedding

# Return as NumPy vector

# Function for semantic search
def semantic_search(query, top_n=3):
    query_embedding = generate_embedding(query)
    similarities = []
    for idx, row in df.iterrows():
        desc_embedding = generate_embedding(row['processed_description'])
        similarity = cosine_similarity(query_embedding, desc_embedding)[0][0]
        similarities.append((row['Job Title'], similarity, row['Job Description']))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Example usage
query = "Sales Manager with experience in B2B sales and team leadership"
results = semantic_search(query)
for title, similarity, description in results:
    print(f"Title: {title}, Similarity: {similarity:.2f}")