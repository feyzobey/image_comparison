import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained VGG16 model with locally downloaded weights
model = VGG16(
    weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    include_top=False,
    input_shape=(224, 224, 3),
)


# Define a function to preprocess images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Load and preprocess the images you want to compare
image1_path = "images/watch1.webp"
image2_path = "images/watch2.jpeg"
image1 = preprocess_image(image1_path)
image2 = preprocess_image(image2_path)

# Extract features from the images using the VGG16 model
features1 = model.predict(image1)
features2 = model.predict(image2)

# Calculate the cosine similarity between the feature vectors
similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
similarity_percentage = similarity * 100

# Set a threshold to determine if the images are similar
threshold = 95  # Adjust the threshold as needed

print(f"Similarity Percentage: {similarity_percentage:.2f}%")
if similarity_percentage >= threshold:
    print("The images are similar.")
else:
    print("The images are not very similar.")
