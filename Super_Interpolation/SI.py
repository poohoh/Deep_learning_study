import cv2
import numpy as np

THETA = 15
LAMBDA_VALUE = 1

# Generate 3x3 LR patch y and 2x2 HR patch x pairs
def get_patch_pairs(hr_img, lr_img):
    lr_patches = []
    hr_patches = []
    for i in range(lr_img.shape[0] - 2):  # height
        for j in range(lr_img.shape[1] - 2):  # width
            lr_patch = []
            hr_patch = []
            # LR patch
            for h in range(3):
                for w in range(3):
                    lr_patch.append(lr_img[i + h][j + w])

            # HR patch
            for h in range(2):
                for w in range(2):
                    hr_patch.append(hr_img[(i + 1) * 2 + h][(j + 1) * 2 + w])

            lr_patches.append(np.array(lr_patch))
            hr_patches.append(np.array(hr_patch))

    return np.array(hr_patches), np.array(lr_patches)

# Generate 2x2 LR sub-patches
def get_sub_patches(lr_patch):
    lr_sub_patches = []
    lr_patch = lr_patch.reshape((3, 3))
    for i in range(2):
        for j in range(2):
            lr_sub_patch = lr_patch[i:i+2, j:j+2]
            lr_sub_patches.append(lr_sub_patch)

    return np.array(lr_sub_patches)

def calculate_gradient(sub_patch):
    sub_patch = sub_patch.reshape(2, 2)
    h_filter = np.array([1, -1, 1, -1]).reshape(2,2)
    v_filter = np.array([1, 1, -1, -1]).reshape(2,2)

    h_gradient = np.sum(sub_patch * h_filter)
    v_gradient = np.sum(sub_patch * v_filter)

    return h_gradient, v_gradient

def compute_class(h_gradients, v_gradients):
    cls_idx = np.zeros(4, dtype=int)
    for i, (h_gradient, v_gradient) in enumerate(zip(h_gradients, v_gradients)):
        m = np.sqrt(h_gradient**2 + v_gradient**2)
        d = np.arctan(h_gradient / v_gradient) if v_gradient != 0 else 3  # v_gradient가 0이면 기울기 값을 무한으로 간주
        d = np.degrees(d)

        if m < THETA:
            cls_idx[i] = 0
            continue

        if -22.5 <= d < 22.5 or 157.5 <= d < 202.5:
            cls_idx[i] = 1
        elif 22.5 <= d < 67.5 or 202.5 <= d < 247.5:
            cls_idx[i] = 2
        elif 67.5 <= d < 112.5 or 247.5 <= d < 292.5:
            cls_idx[i] = 3
        else:
            cls_idx[i] = 4

    cls = 1*cls_idx[0] + 5*cls_idx[1] + 5**2 * cls_idx[2] + 5**3 * cls_idx[3]

    return cls


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

    print(lin_map)
    print()

img = cv2.imread('./Data/Training/t1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cb, cr = cv2.split(img)

# merged = cv2.merge([y, cb, cr])
# merged = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

training(y)