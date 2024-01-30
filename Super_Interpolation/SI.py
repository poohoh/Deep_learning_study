import cv2

def training(hr_img):
    # Generate LR images L
    blur_img = cv2.blur(hr_img, (3, 3))
    lr_img = cv2.resize(blur_img, (hr_img.shape[1]//2, hr_img.shape[0]//2))

    # Generate 3x3 LR patch y and 2x2 HR patch x pairs
    lr_patches = []
    hr_patches = []
    for i in range(lr_img.shape[0]-2):  # height
        for j in range(lr_img.shape[1]-2):  # width
            lr_patch = []
            hr_patch = []
            # LR patch
            for h in range(3):
                for w in range(3):
                    lr_patch.append(lr_img[i+h][j+w])

            # HR patch
            for h in range(2):
                for w in range(2):
                    hr_patch.append(hr_img[(i+1)*2 + h][(j+1)*2 + w])

            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)

img = cv2.imread('./Data/Training/t1.bmp')
training(img)