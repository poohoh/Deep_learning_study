import time

from training import *
from up_scaling import up_scaling
import os

def main():
    data_dir = './Data/Training'
    images = os.listdir(data_dir)
    hr_images = []
    for image in images[:30]:
        img = cv2.imread(os.path.join(data_dir, image))
        hr_images.append(img)

    lin_map = training(hr_images)

    empty = [i for i, arr in enumerate(lin_map) if len(arr) == 0]
    print(empty)

    # up-scaling
    test_img = cv2.imread('./Data/Testing/Child_gnd.bmp')
    result = up_scaling(test_img, lin_map)

    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)

    cv2.imwrite(os.path.join(result_dir, f'up_scaling_result_{time.time()}.png'), result)
    cv2.imwrite(os.path.join(result_dir, f'origin_{time.time()}.png'), test_img)

if __name__=='__main__':
    main()