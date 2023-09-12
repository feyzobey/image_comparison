import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
# Load a pre-trained model (e.g., ResNet-18)
base_model = models.resnet18(pretrained=True)
# Extract the feature extractor (without the final classification layer)
feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
# Find the number of output features (in_features) of the feature extractor
in_features = base_model.fc.in_features  # Access in_features here

# Define a Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, in_features):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(in_features, 128)

    def forward_one(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size()[0], -1)
        output = self.fc(features)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

siamese_model = SiameseNetwork(in_features)

# Define an image similarity function
def image_similarity(image1, image2):
    # Preprocess images (resize, normalize, etc.)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image1 = preprocess(image1).unsqueeze(0)
    image2 = preprocess(image2).unsqueeze(0)

    # Pass images through the Siamese network
    with torch.no_grad():
        output1, output2 = siamese_model(image1, image2)
    
    # Compute cosine similarity between the embeddings
    similarity = nn.functional.cosine_similarity(output1, output2)
    
    return similarity.item()  # Similarity score between -1 and 1

# Load two images
image1_path = "images/at.jpeg"
image2_path = "images/com.jpeg"
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Compare the images and get the similarity score
similarity_score = image_similarity(image1, image2)

print(f"Similarity Score: {similarity_score}")
