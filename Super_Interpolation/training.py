import cv2

from function import *

def training(hr_images):
    # class별 cluster
    hr_cluster = [[] for _ in range(625)]  # n x 4
    lr_cluster = [[] for _ in range(625)]  # n x 9

    for num, hr_image in enumerate(hr_images):
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)
        hr_img, cb, cr = cv2.split(hr_image)

        # Generate LR images L
        blur_img = cv2.blur(hr_img, (3, 3))
        lr_img = cv2.resize(blur_img, (hr_img.shape[1]//2, hr_img.shape[0]//2), interpolation=cv2.INTER_CUBIC)

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
        for i in range(625):
            hr_cluster[i] = hr_cluster[i] + list(hr_patches[np.where(cls==i)])
            lr_cluster[i] = lr_cluster[i] + list(lr_patches[np.where(cls==i)])

        print(f'hr_img[{num}] completed')

    # compute linear mappings matrix for each cluster of EO classes
    lin_map = []
    for i in range(625):
        # 구하고자 하는 행렬의 크기는 9 x 4
        # M = XY^T(YY^T + lI)^(-1), X: hr, Y: lr
        # M = (Y^TY + lI)^(-1)Y^TX
        x = np.array(hr_cluster[i])
        y = np.array(lr_cluster[i])
        if len(x) > 0:
            m = np.linalg.inv(y.T @ y + LAMBDA_VALUE * np.eye(y.shape[1])) @ y.T @ x
            # m = x @ y.T @ np.linalg.inv(y @ y.T + LAMBDA_VALUE * np.eye(y.shape[0]))
        else:
            m = []
        lin_map.append(m)

    return lin_map