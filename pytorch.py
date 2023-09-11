# not tested yet, download model first( 1.2GB ) and put it in the same folder as this file 
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Load the pre-trained ViT-Large model and feature extractor
model_name = "google/vit-large-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load and preprocess the images
image1_path = "path_to_image1.jpg"
image2_path = "path_to_image2.jpg"

def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

image1_inputs = preprocess_image(image1_path)
image2_inputs = preprocess_image(image2_path)

# Perform image comparison by computing embeddings
with torch.no_grad():
    embeddings1 = model(**image1_inputs).last_hidden_state.mean(dim=1)  # Extract embeddings for image1
    embeddings2 = model(**image2_inputs).last_hidden_state.mean(dim=1)  # Extract embeddings for image2

# Compute the cosine similarity between the image embeddings
cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

# Define a threshold for similarity
similarity_threshold = 0.9  # You can adjust this threshold as needed

# Compare the images based on the cosine similarity
if cosine_similarity.item() >= similarity_threshold:
    print("The images are similar.")
else:
    print("The images are dissimilar.")
