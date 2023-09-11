import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the ViT-Large model
model_url = "https://tfhub.dev/google/vit-large-patch16/imagenet21k/2"
vit_model = hub.load(model_url)

# Function to preprocess images for the ViT model
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((384, 384))  # Resize to match the input size of ViT
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to the [0, 1] range
    return img

# Load and preprocess the two images to compare
image1_path = "images/kartus1.jpeg"
image2_path = "images/kartus3.jpeg"
image1 = preprocess_image(image1_path)
image2 = preprocess_image(image2_path)

# Convert images to batch format (add batch dimension)
image1_batch = tf.expand_dims(image1, axis=0)
image2_batch = tf.expand_dims(image2, axis=0)

# Perform inference with the ViT model
features1 = vit_model(image1_batch)
features2 = vit_model(image2_batch)

# Calculate cosine similarity between the feature vectors
cosine_similarity = tf.keras.losses.cosine_similarity(features1, features2, axis=-1)
similarity_score = 1 - cosine_similarity  # Convert to similarity score

# Set a threshold to determine if the images are similar
threshold = 0.9  # Adjust the threshold as needed

# Compare the similarity score to the threshold
if similarity_score >= threshold:
    print("The images are similar.")
else:
    print("The images are not very similar.")
