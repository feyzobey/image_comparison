from skimage import io, transform
from skimage.metrics import structural_similarity as ssim

# Load the two images you want to compare
image1_path = "images/watch3.webp"
image2_path = "images/watch4.jpeg"
image1 = io.imread(image1_path, as_gray=True)
image2 = io.imread(image2_path, as_gray=True)

# Resize the images to have the same dimensions
if image1.shape != image2.shape:
    common_shape = (
        min(image1.shape[0], image2.shape[0]),
        min(image1.shape[1], image2.shape[1]),
    )
    image1 = transform.resize(image1, common_shape)
    image2 = transform.resize(image2, common_shape)

# Calculate the SSIM between the two images
ssim_score = ssim(
    image1, image2, data_range=image2.max() - image2.min(), multichannel=True
)

# Calculate the similarity percentage (convert SSIM to percentage)
similarity_percentage = ssim_score * 100

print(f"Similarity Percentage: {similarity_percentage:.2f}%")

# You can set a threshold to determine if the images are similar or not
threshold = 98  # Adjust the threshold as needed

if similarity_percentage >= threshold:
    print("The images are similar.")
else:
    print("The images are not very similar.")
