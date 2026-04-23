import clip
import torch
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from IPython.display import display

# Load the model and preprocess function from CLIP
model, preprocess = clip.load("ViT-B/32")
model.eval()

# Load the embeddings and metadata
embeddings = torch.load('/usr/local/datasetsDir/images-and-descriptions/embeddings.pt')
image_features = embeddings['image_features']
text_features = embeddings['text_features']
image_paths = embeddings['image_paths']
descriptions = embeddings['descriptions']

# Function to search images and descriptions by image
def search_by_image(query_image_path, image_features, text_features, image_paths, descriptions):
    image_input = preprocess(Image.open(query_image_path)).unsqueeze(0) # After preprocessing the image we unsqueeze to add a batch dimension. Image encoder expects input images to have a shape of (batch_size, channels=3, height=224, width=224).
    with torch.no_grad():
        query_feature = model.encode_image(image_input)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)

    similarities = cosine_similarity(query_feature, text_features)

    top_match_idxs = top_match_idxs = similarities[0].argsort()[-5:][::-1]
    results = [(image_paths[idx], descriptions[idx], similarities[0][idx].item()) for idx in top_match_idxs]

    return results

# Test
query_image_path = '/usr/local/datasetsDir/images-and-descriptions/queries/girlwithorangesliceoneyes.jpg'
print("Query Image:")
display(Image.open(query_image_path))
print("Finding similar results, please wait...")
results = search_by_image(query_image_path, image_features, text_features, image_paths, descriptions)
for idx, (image_path, description, similarity) in enumerate(results):
    print(f"Match {idx + 1}:")
    display(Image.open(image_path))
    print(f"Description: {description} (Similarity: {similarity:.4f})")
    print("------------")