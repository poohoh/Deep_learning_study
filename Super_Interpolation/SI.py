import cv2
import numpy as np

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

    print(lr_patch)
    print(lr_sub_patches)

    return np.array(lr_sub_patches)

def calculate_gradient(sub_patch):
    sub_patch = sub_patch.reshape(-1)
    h_filter = np.array([1, -1, -1, 1])
    v_filter = np.array([1, 1, -1, -1])

    h_gradient = sub_patch * h_filter
    h_gradient = np.sum(h_gradient, axis=0)
    v_gradient = sub_patch * v_filter
    v_gradient = np.sum(v_gradient, axis=0)

    return h_gradient, v_gradient

def training(hr_img):
    # Generate LR images L
    blur_img = cv2.blur(hr_img, (3, 3))
    lr_img = cv2.resize(blur_img, (hr_img.shape[1]//2, hr_img.shape[0]//2))

    # Generate 3x3 LR patch y and 2x2 HR patch x pairs
    hr_patches, lr_patches = get_patch_pairs(hr_img, lr_img)

    print(hr_patches.shape)
    print(lr_patches.shape)

    # Each LR external patch
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


img = cv2.imread('./Data/Training/t1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cb, cr = cv2.split(img)

# merged = cv2.merge([y, cb, cr])
# merged = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

training(y)