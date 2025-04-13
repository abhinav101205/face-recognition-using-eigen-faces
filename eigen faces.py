import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA as ipc
import cv2
import os
from sklearn.metrics.pairwise import euclidean_distances

# Load training images
folderpath = "/content/sample_data/training"
images = []
dimensions = []
train_image_names = []

for filename in os.listdir(folderpath):
    img = cv2.imread(os.path.join(folderpath, filename))
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dimensions.append(gray.shape)
        images.append(gray)
        train_image_names.append(filename)

print(f"Loaded {len(images)} training images.")

#Display all training images in a grid
num_images = len(images)
cols = 6
rows = (num_images + cols - 1) // cols

plt.figure(figsize=(15, rows * 2))
for i, img in enumerate(images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv2.resize(img, (64, 64)), cmap='gray')
    plt.title(f"{train_image_names[i]}", fontsize=8)
    plt.axis('off')
plt.suptitle("All Training Images", fontsize=16)
plt.tight_layout()
plt.show()


# Preprocess training images
if len(images) > 0:
    target_shape = images[0].shape
    resized_images = [cv2.resize(img, (target_shape[1], target_shape[0])) for img in images]
    flattened_images = [img.flatten() for img in resized_images]
    image_matrix = np.array(flattened_images)

    image_stack = np.stack(resized_images, axis=0)
    mean_image = np.mean(image_stack, axis=0)

    plt.figure(figsize=(8, 8))
    plt.imshow(mean_image, cmap='gray')
    plt.title("Mean Image of All Grayscale Images")
    plt.axis('off')
    plt.show()

# PCA training
n_components = min(max(10, image_matrix.shape[0]), 100)
ipca = ipc(n_components=n_components, batch_size=10)
ipca.fit(image_matrix)

print(f"PCA fitted with {n_components} components.")

# Load test images
test_folderpath = "/content/sample_data/test"
test_images = []
test_image_names = []

for filename in os.listdir(test_folderpath):
    img = cv2.imread(os.path.join(test_folderpath, filename))
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (target_shape[1], target_shape[0]))
        test_images.append(gray_resized.flatten())
        test_image_names.append(filename)

print(f"Loaded {len(test_images)} test images.")

# Project into PCA space
train_pca_projection = ipca.transform(image_matrix)
test_pca_projection = ipca.transform(np.array(test_images))

# Match test images to training images
for i, test_img_pca in enumerate(test_pca_projection):
    distances = euclidean_distances([test_img_pca], train_pca_projection)
    closest_match_index = np.argmin(distances)

    reconstructed_test_img = ipca.inverse_transform([test_img_pca])
    reconstructed_test_img_reshaped = reconstructed_test_img.reshape(target_shape)

    # Plot the original test image, its eigenface, and the closest match
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(test_images[i].reshape(target_shape), cmap='gray')
    plt.title(f"Original Test Image\n{test_image_names[i]}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_test_img_reshaped, cmap='gray')
    plt.title("Eigenface")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(images[closest_match_index], cmap='gray')
    plt.title(f"Closest Match: {train_image_names[closest_match_index]}")
    plt.axis('off')

    plt.suptitle(f"Test Image {i+1}: Closest Match Found", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Test Image {i+1} is closest to Training Image {closest_match_index + 1} ({train_image_names[closest_match_index]})")
