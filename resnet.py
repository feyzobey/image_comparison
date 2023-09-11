import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained ResNet-50 model with locally downloaded weights
model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# List of image file paths to compare against each other
image_paths_to_compare = ['images/watch1.webp', 'images/watch2.jpeg']

# Directory to save the converted JPEG images
output_directory = 'converted_images'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Convert each image to JPEG format and save it
converted_image_paths = []
for image_path in image_paths_to_compare:
    try:
        # Check if the image is already in JPEG format
        if not image_path.lower().endswith('.jpeg'):
            # Open the image
            img = Image.open(image_path)

            # Convert to JPEG format with a specified quality (adjust as needed)
            img = img.convert('RGB')
            jpeg_path = os.path.join(output_directory, os.path.splitext(os.path.basename(image_path))[0] + '.jpeg')
            img.save(jpeg_path, 'JPEG', quality=95)
            
            converted_image_paths.append(jpeg_path)
            print(f"Converted {image_path} to JPEG format: {jpeg_path}")
        else:
            converted_image_paths.append(image_path)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# Initialize a similarity matrix to store similarity scores
num_images = len(converted_image_paths)
similarity_matrix = np.zeros((num_images, num_images))

# Compare each pair of images and fill in the similarity matrix
for i in range(num_images):
    for j in range(i+1, num_images):
        image_path1 = converted_image_paths[i]
        image_path2 = converted_image_paths[j]

        # Load and preprocess the images to compare
        image1 = preprocess_image(image_path1)
        image2 = preprocess_image(image_path2)

        # Extract features from the images using the ResNet-50 model
        features1 = model.predict(image1)
        features2 = model.predict(image2)

        # Calculate the cosine similarity between the feature vectors
        similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

        # Store the similarity score in the matrix (symmetric matrix)
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

# Set a threshold to determine if the images are similar
threshold = 0.95  # Adjust the threshold as needed

# Print the similarity matrix and identify similar images
for i in range(num_images):
    for j in range(i+1, num_images):
        image1_path = converted_image_paths[i]
        image2_path = converted_image_paths[j]
        similarity_score = similarity_matrix[i][j]
        
        print(f"Similarity between {image1_path} and {image2_path}: {similarity_score:.4f}")
        
        if similarity_score >= threshold:
            print(f"The images {image1_path} and {image2_path} are similar.")
