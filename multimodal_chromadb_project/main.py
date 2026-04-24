import os
import warnings

import chromadb
import pandas as pd
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from IPython.display import Image as IPImage
from IPython.display import display

warnings.filterwarnings("ignore")


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "data", "image_descriptions.csv")
IMAGES_FOLDER = os.path.join(BASE_DIR, "data", "images")
QUERY_IMAGE_PATH = os.path.join(BASE_DIR, "queries", "girlwithorangesliceoneyes.jpg")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")


# -----------------------------
# Configure embedding model
# -----------------------------
model_name = "ViT-B-32"
embedding_function = OpenCLIPEmbeddingFunction(model_name=model_name)
data_loader = ImageLoader()


# -----------------------------
# Create Chroma persistent client
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection_name = "multimodal_embeddings_collection"

collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
    data_loader=data_loader,
)


# -----------------------------
# Load and prepare dataset
# -----------------------------
df = pd.read_csv(CSV_FILE)

image_paths = []
image_ids = []
descriptions = []
description_ids = []

for _, row in df.iterrows():
    desc_id = str(row.iloc[0])
    description = str(row.iloc[1])

    for file_name in os.listdir(IMAGES_FOLDER):
        if file_name.startswith(f"{desc_id}_") and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(IMAGES_FOLDER, file_name)

            image_ids.append(file_name)
            image_paths.append(image_path)
            descriptions.append(description)
            description_ids.append(desc_id)
            break


# -----------------------------
# Add images and descriptions
# -----------------------------
for img_id, img_path, desc_id, desc in zip(
    image_ids, image_paths, description_ids, descriptions
):
    collection.add(
        ids=[f"image_{img_id}"],
        uris=[img_path],
        metadatas=[{"image_uri": img_path, "description": desc, "type": "image"}],
    )

    collection.add(
        ids=[f"text_{desc_id}"],
        documents=[desc],
        metadatas=[{"image_uri": img_path, "description": desc, "type": "text"}],
    )


print(f"Added {len(image_ids)} images and {len(description_ids)} descriptions to ChromaDB.")


# -----------------------------
# Query by text
# -----------------------------
query_text = "vitamin C fruits"

text_query_results = collection.query(
    query_texts=[query_text],
    n_results=5,
)

print("\nText query:", query_text)
print("*********** Text Query Results ************")

for metadata, distance in zip(
    text_query_results["metadatas"][0],
    text_query_results["distances"][0],
):
    print(f"\nDescription: {metadata['description']}")
    print(f"Distance: {distance:.4f}")
    display(IPImage(filename=metadata["image_uri"]))


# -----------------------------
# Query by image
# -----------------------------
if os.path.exists(QUERY_IMAGE_PATH):
    image_query_results = collection.query(
        query_uris=[QUERY_IMAGE_PATH],
        n_results=5,
    )

    print("\nImage query:")
    print(f"Query image path: {QUERY_IMAGE_PATH}")
    print("*********** Image Query Results ************")

    for metadata, distance in zip(
        image_query_results["metadatas"][0],
        image_query_results["distances"][0],
    ):
        print(f"\nDescription: {metadata['description']}")
        print(f"Distance: {distance:.4f}")
        print(f"Image path: {metadata['image_uri']}")
else:
    print("\nQuery image not found. Add your query image inside the queries folder.")