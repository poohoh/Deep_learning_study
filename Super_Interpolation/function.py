import numpy as np
import cv2

THETA = 15
LAMBDA_VALUE = 1

# Generate 3x3 LR patch y and 2x2 HR patch x pairs
def get_patch_pairs(hr_img, lr_img):
    lr_patches = []
    hr_patches = []
    for i in range(lr_img.shape[0] - 2):  # height
        for j in range(lr_img.shape[1] - 2):  # width
            lr_patch = np.array(lr_img[i:i+3, j:j+3]).reshape(9).tolist()
            hr_patch = np.array(hr_img[(i+1)*2:(i+1)*2+2, (j+1)*2:(j+1)*2+2]).reshape(4).tolist()

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

def get_patches(lr_img):
    lr_patches = []
    for i in range(lr_img.shape[0] - 2):  # height
        for j in range(lr_img.shape[1] - 2):  # width
            lr_patch = np.array(lr_img[i:i+3, j:j+3]).reshape(9).tolist()
            lr_patches.append(np.array(lr_patch))

    return np.array(lr_patches)