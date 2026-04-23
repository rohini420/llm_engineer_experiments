# 🔍 Semantic Job Description Search using BERT

> Match jobs by **meaning**, not just keywords — powered by BERT embeddings and cosine similarity.

---

## What It Does

This project performs **semantic search** on job descriptions. Instead of relying on keyword matching, it understands the *intent* behind a query and returns the most contextually relevant job titles using a pretrained BERT model.

---

## Features

- 📂 Loads job descriptions from a CSV file
- 🧹 Cleans and preprocesses text (tokenization, stopword removal, lemmatization)
- 🤖 Uses `bert-base-uncased` to generate sentence embeddings via the `[CLS]` token
- 📐 Computes cosine similarity between your query and job descriptions
- 🏆 Returns the top-N most semantically similar job titles

---

## Project Structure

```
project/
│
├── job_title_des.csv          # Input dataset (not included in repo)
├── semantic_search.py         # Main script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## How It Works

### 1. Load and Prepare the Dataset
The script reads a CSV file containing job titles and descriptions, using the first 20 rows for demonstration.

### 2. Text Preprocessing
Each job description is cleaned by:
- Converting to lowercase
- Tokenizing
- Removing stopwords and punctuation
- Lemmatizing words

A new column `processed_description` is added to the DataFrame.

### 3. BERT Embedding Generation
- `BertTokenizer` converts text into tokens
- `BertModel` generates embeddings
- The `[CLS]` token output represents the entire sentence as a single vector

### 4. Semantic Search
For a given query:
1. Generate its BERT embedding
2. Compare it against each job description embedding using cosine similarity
3. Sort results by similarity score
4. Return the top matches

---

## Example

```python
query = "Sales Manager with experience in B2B sales and team leadership"
results = semantic_search(query)
```

**Output:**
```
Title: Senior Sales Manager,    Similarity: 0.87
Title: B2B Sales Lead,          Similarity: 0.82
Title: Sales Team Supervisor,   Similarity: 0.79
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources

Run this once inside Python:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

```bash
python semantic_search.py
```

Modify the `query` variable inside the script to test different job searches.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `torch` | PyTorch backend for BERT |
| `transformers` | BERT tokenizer and model |
| `scikit-learn` | Cosine similarity computation |
| `nltk` | Text preprocessing |

All dependencies are listed in `requirements.txt`.

---

## Notes

- 📁 The dataset file (`job_title_des.csv`) is **not included** in this repository — bring your own.
- ⏳ BERT models may take a moment to load depending on your hardware.
- ⚡ For larger datasets, consider **caching embeddings** to avoid recomputing them on every run.

---

## License

This project is open source. Feel free to use, modify, and distribute.
