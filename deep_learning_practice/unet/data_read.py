## 딥러닝 할껀데, 실습만 합니다. - 002 UNet

## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

##
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 랜덤 프레임 인덱스
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## 기존 파일 삭제
def delete_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)

delete_files(dir_save_train)
delete_files(dir_save_val)
delete_files(dir_save_test)

## train data 저장
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[offset_nframe + i])
    img_input.seek(id_frame[offset_nframe + i])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_train, f'input_{i:03d}.npy'), input_)

## val data 저장
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[offset_nframe + i])
    img_input.seek(id_frame[offset_nframe + i])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_val, f'input_{i:03d}.npy'), input_)

## test data 저장
offset_nframe += nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[offset_nframe + i])
    img_input.seek(id_frame[offset_nframe + i])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_test, f'input_{i:03d}.npy'), input_)

## 생성된 데이터 출력
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()








































