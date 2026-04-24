# Multimodal Embeddings Search with ChromaDB

A mini AI project that demonstrates **multimodal semantic search** using **images + text descriptions** stored in **ChromaDB**.

This project uses **OpenCLIP embeddings** to place both images and text in the same vector space, allowing you to:

- 🔍 Search images using text queries
- 🖼️ Search similar images using another image
- 🗄️ Store multimodal embeddings in a vector database
- 🧠 Understand real-world Retrieval Augmented Generation (RAG) foundations

---

## Project Overview

Traditional databases store rows and columns. Vector databases store **embeddings** — numerical representations of data that capture semantic meaning.

In this project:

- Images are converted into vectors
- Text descriptions are converted into vectors
- Both are stored inside ChromaDB
- Similarity search is performed using cosine distance

---

## Example Use Cases

### Text Query

```text
Query: "vitamin C fruits"

Results:
1. Orange
2. Orange slice
3. Apple
4. Banana
```

### Image Query

Upload or query an image to retrieve visually similar images from the database.

---

## Folder Structure

```text
multimodal_chromadb_project/
├── main.py
├── README.md
├── requirements.txt
├── chroma_db/
├── data/
│   ├── image_descriptions.csv
│   └── images/
└── queries/
    └── girlwithorangesliceoneyes.jpg
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| ChromaDB | Vector database |
| OpenCLIP | Multimodal embeddings |
| Pandas | Data handling |
| PyTorch | Model inference |
| Pillow | Image processing |

---

## Installation

### 1. Clone / Open Project

```bash
cd multimodal_chromadb_project
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install chromadb pandas torch pillow scikit-learn open_clip_torch ipython
```

---

## Run the Project

```bash
python main.py
```

---

## How It Works

### Step 1 — Load Dataset

Reads `image_descriptions.csv` and image files from the `data/` directory.

### Step 2 — Generate Embeddings

Uses the OpenCLIP model `ViT-B-32` to generate embeddings for both images and their text descriptions.

### Step 3 — Store in ChromaDB

Embeddings are stored in a persistent vector database at `chroma_db/`.

### Step 4 — Query

**Text search:**
```python
query_text = "vitamin C fruits"
```

**Image search:**
```python
query_uris = [query_image]
```

---

## Sample Output

```text
Added 30 images and 30 descriptions to ChromaDB.

Text Query: vitamin C fruits

1. Orange
2. Orange slice
3. Apple
4. Banana
```

---

## Key Concepts Learned

- Vector Databases
- Embeddings & Embedding Models
- Multimodal AI
- Semantic Search
- Cross-modal Retrieval
- Cosine Similarity
- ChromaDB Collections
- OpenCLIP Models

---

## Real-World Applications

- 🛍️ E-commerce visual search
- 🏥 Medical image retrieval
- 📄 Document + image search
- 🤖 AI recommendation engines
- 👗 Fashion & retail similarity search

---

## Future Improvements

- [ ] FastAPI layer for REST endpoints
- [ ] Streamlit UI for interactive search
- [ ] Custom image upload support
- [ ] Hybrid search (keyword + vector)
- [ ] Metadata filtering
- [ ] AWS deployment
- [ ] RAG chatbot integration

---

## Resume Project Description

> Built a multimodal semantic search system using **ChromaDB** and **OpenCLIP** embeddings to index images and text descriptions in a shared vector space, enabling cross-modal retrieval through text-to-image and image-to-image similarity search.

---

## Author

Your Name

## License

[MIT](LICENSE)
