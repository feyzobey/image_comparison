# from skimage.metrics import structural_similarity
# from skimage.transform import resize
# import cv2

# Ensure both images have the same dimensions
# img1 = resize(
#     img1, (img2.shape[0], img2.shape[1]), anti_aliasing=True, preserve_range=True
# )


# img1 = cv2.imread("images/image3_resized.jpg", 0)
# img2 = cv2.imread("images/image4_resized.jpg", 0)
# # Calculate SSIM with data_range specified
# ssim_value, diff = structural_similarity(img1, img2, full=True, data_range=255)


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def resize_image(image, target_size):
    return cv2.resize(image, target_size)


def image_similarity(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Check if the images were loaded successfully
    if image1 is None or image2 is None:
        raise ValueError("Could not load one or both of the images.")

    # Resize images to have the same dimensions
    # target_size = (image1.shape[1], image1.shape[0])  # Use the dimensions of the first image
    # image2_resized = resize_image(image2, target_size)

    # Convert images to grayscale (SSIM requires grayscale images)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM score
    ssim_score = ssim(image1_gray, image2_gray, data_range=255)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    return ssim_score, mse


if __name__ == "__main__":
    # Replace these with the paths to your images
    image1_path = "images/image9.jpg"
    image2_path = "images/image10.jpg"
    im1 = Image.open(image1_path)
    im2 = Image.open(image2_path)
    print(im1.size, im2.size)

    # if im1.size[0] * im1.size[1] < im2.size[0] * im2.size[1]:
    #     im2 = im2.resize(im1.size)
    # else:
    #     im1 = im1.resize(im2.size)

    # image1_path = image1_path.split(".")[0] + "_resized." + image1_path.split(".")[1]
    # image2_path = image2_path.split(".")[0] + "_resized." + image2_path.split(".")[1]
    # im1.save(image1_path)
    # im2.save(image2_path)
    # im1.show()
    # im2.show()
    try:
        ssim_score, mse = image_similarity(image1_path, image2_path)
        print(f"SSIM Score: {ssim_score}")
        print(f"Mean Squared Error (MSE): {mse}")
    except ValueError as e:
        print(e)
