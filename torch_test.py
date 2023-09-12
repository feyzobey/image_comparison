import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import timm

# Load a pre-trained VIT model (using timm)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Load and preprocess the images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match VIT's input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Load and preprocess the two images you want to compare
image1_path = 'images/kartus2.jpeg'
image2_path = 'images/kartus3.jpeg'
image1 = preprocess_image(image1_path)
image2 = preprocess_image(image2_path)

# Get the feature embeddings from the VIT model
with torch.no_grad():
    features1 = model(image1)
    features2 = model(image2)

# Calculate the cosine similarity between the feature embeddings
cosine_similarity = F.cosine_similarity(features1, features2)

# Convert cosine similarity to a percentage (0% to 100%)
similarity_percentage = (cosine_similarity.item() + 1) / 2 * 100

# Print the similarity percentage
print(f"Similarity Percentage: {similarity_percentage:.2f}%")
