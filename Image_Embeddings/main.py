import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained ResNet model
resnet_model = models.resnet18(pretrained=True)
# Remove the final fully connected layer
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
# Set the model to evaluation mode
resnet_model.eval()
# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    # Define transformations to be applied to the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Apply transformations
    image_tensor = preprocess(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
# Function to generate image embeddings
def generate_image_embedding(image_path):
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    # Forward pass through the model to obtain features
    with torch.no_grad():
        features = resnet_model(image_tensor)
    # Flatten the feature map
    embedding = features.squeeze().numpy()
    return embedding
# Function for semantic search
def semantic_search(query_embedding, image_folder, top_n=2):
    similarities = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            # Generate embedding for the image
            image_embedding = generate_image_embedding(image_path)
            # Compute cosine similarity between query embedding and image embedding
            similarity = cosine_similarity([query_embedding], [image_embedding])[0][0]
            similarities.append((filename, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
