import os.path

import cv2
import random
import numpy as np

# 클러스터 중심을 랜덤으로 생성
def get_center(height, width):
    # spatial
    # x = random.randint(0, width-1)
    # y = random.randint(0, height-1)

    # return np.array((x, y))

    # r, g, b channel
    r, g, b = [random.randint(0, 255) for _ in range(3)]

    return r, g, b

# label update
def update_label(img, labels, centroids):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distances = [np.linalg.norm(img[i][j] - centroids[k]) for k in range(num_centroid)]
            label = distances.index(min(distances))
            labels[i][j] = label

# centroids update
def update_centroids(img, labels, centroids):
    for i in range(len(centroids)):
        indices = np.where(labels==i)
        if len(indices[0]) > 0:  # 클래스에 속하는 점이 있는 경우
            r_center = np.mean(img[np.where(labels==i)][0])
            g_center = np.mean(img[np.where(labels==i)][1])
            b_center = np.mean(img[np.where(labels==i)][2])

            centroids[i] = (r_center, g_center, b_center)

# read image
img = cv2.imread('./airplane.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 클러스터 중심 개수 지정 및 생성
num_centroid = 5
centroids = [get_center(img.shape[0], img.shape[1]) for _ in range(num_centroid)]
labels = np.zeros((316, 316)).astype(np.uint8)

# save
save_dir = './result_k_means'
os.makedirs(save_dir, exist_ok=True)

# visualize
colors = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]).astype(np.uint8)
vis = colors[labels]
save_path = os.path.join(save_dir, f'initial_labels.png')  # save initial labels
cv2.imwrite(save_path, vis)
cv2.imwrite(os.path.join(save_dir, 'origin_image.png'), img)  # save original image

# # update
iter = 30
for i in range(iter):
    # update labels
    update_label(img, labels, centroids)

    # update centroids
    update_centroids(img, labels, centroids)

    print(f'iteration: {i+1}/{iter}')
    print(f'centroids: {centroids}')

    # save labels
    save_path = os.path.join(save_dir, f'iter_{i+1}.png')
    vis = colors[labels]
    cv2.imwrite(save_path, vis)