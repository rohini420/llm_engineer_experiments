import os
import pandas as pd
import clip
import torch
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

# Load the model and preprocess function from CLIP
model, preprocess = clip.load("ViT-B/32")
model.eval()

# Path configurations
images_folder = '/usr/local/datasetsDir/images-and-descriptions/data/images'
csv_file = '/usr/local/datasetsDir/images-and-descriptions/data/image_descriptions.csv'

# Load the CSV file with image IDs and descriptions
df = pd.read_csv(csv_file)

# Prepare lists for image paths and descriptions
image_paths = []
descriptions = []

# Iterate through the CSV file to get image paths and corresponding descriptions
for _, row in df.iterrows():
    image_id = str(row[0])  # Ensure that the ID is a string for matching
    description = row[1]
    # Find the image file corresponding to the image_id
    for file_name in os.listdir(images_folder):
        if file_name.startswith(f"{image_id}_") and file_name.endswith('.png'):
            image_path = os.path.join(images_folder, file_name)
            image_paths.append(image_path)
            descriptions.append(description)
            break

# Function to preprocess images and text descriptions
def preprocess_data(image_paths, descriptions):
    images = [preprocess(Image.open(image_path)) for image_path in image_paths]
    text_inputs = [clip.tokenize(description, truncate=True) for description in descriptions]
    return torch.stack(images), torch.cat(text_inputs)

# Preprocess images and descriptions
images, text_inputs = preprocess_data(image_paths, descriptions)

# Generate embeddings
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text_inputs)
# Normalize the features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Print embeddings
print("Image embeddings: ", image_features)
print("Text embeddings: ", text_features)

# Save the embeddings and metadata for later use
torch.save({
    'image_features': image_features,
    'text_features': text_features,
    'image_paths': image_paths,
    'descriptions': descriptions
}, '/usr/local/datasetsDir/images-and-descriptions/embeddings.pt')

print("Embeddings generated and saved successfully.")