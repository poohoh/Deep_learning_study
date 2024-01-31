import cv2
from function import *

def training(hr_img):
    # Generate LR images L
    blur_img = cv2.blur(hr_img, (3, 3))
    lr_img = cv2.resize(blur_img, (hr_img.shape[1]//2, hr_img.shape[0]//2))

    # Generate 3x3 LR patch y and 2x2 HR patch x pairs
    hr_patches, lr_patches = get_patch_pairs(hr_img, lr_img)

    # Each LR external patch
    cls = []
    for lr_patch in lr_patches:
        # get sub patches
        lr_sub_patches = get_sub_patches(lr_patch)

        # Each 2x2 LR sub-patches
        h_gradients = []
        v_gradients = []
        for sub_patch in lr_sub_patches:
            h_gradient, v_gradient = calculate_gradient(sub_patch)
            h_gradients.append(h_gradient)
            v_gradients.append(v_gradient)

        # compute EO class index
        cls.append(compute_class(h_gradients, v_gradients))

    # make cluster
    cls = np.array(cls)
    # 인덱스의 순서대로 lr, hr 클러스터 생성
    hr_cluster = []  # n x 4
    lr_cluster = []  # n x 9
    for i in range(625):
        hr_cluster.append(hr_patches[np.where(cls==i)])
        lr_cluster.append(lr_patches[np.where(cls==i)])

    # compute linear mappings matrix for each cluster of EO classes
    lin_map = []
    for i in range(625):
        # 구하고자 하는 행렬의 크기는 9 x 4
        # M = XY^T(YY^T + lI)^(-1), X: hr, Y: lr
        x = hr_cluster[i]
        y = lr_cluster[i]
        if len(x) > 0:
            m = np.linalg.inv(y.T @ y + LAMBDA_VALUE * np.eye(y.shape[1])) @ y.T @ x
        else:
            m = []
        lin_map.append(m)