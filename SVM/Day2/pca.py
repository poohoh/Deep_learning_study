import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

image_dir = './eigenface_test'
image_data = []

for file in os.listdir(image_dir):
    if file.endswith('.png'):
        # image load
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # add list
        image_array = image.flatten()
        image_data.append(image_array)

# list to numpy
image_data = np.array(image_data)

# PCA
pca = PCA(n_components=20)
pca.fit(image_data)
image_pca = pca.transform(image_data)

# visualize principal components
principal_components = pca.components_
plt.figure(figsize=(15, 10))

n_components_per_row = 6
for i in range(pca.n_components_):
    plt.subplot(4, 5, i + 1)

    pc_image = principal_components[i].reshape((45, 40))
    plt.imshow(pc_image, cmap='gray', aspect='auto')

    plt.title(f'PC {i+1}')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('Principal Components Visualization')
plt.show()

# visualize projection result
restored_image = pca.inverse_transform(image_pca)
plt.figure(figsize=(15, 10))

n_components_per_row = 6
for i in range(len(image_data)):
    plt.subplot(4, 5, i + 1)

    pc_image = restored_image[i].reshape((45, 40))
    plt.imshow(pc_image, cmap='gray', aspect='auto')

    plt.title(f'PC {i+1}')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('Principal Components Visualization')
plt.show()