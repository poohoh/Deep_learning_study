from training import *
import os

# img = cv2.imread('./Data/Training/t1.bmp')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

data_dir = './Data/Training'
images = os.listdir(data_dir)
hr_images = []
for image in images[:10]:
    img = cv2.imread(os.path.join(data_dir, image))
    hr_images.append(img)

lin_map = training(hr_images)

# merged = cv2.merge([y, cb, cr])
# merged = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

print(lin_map)