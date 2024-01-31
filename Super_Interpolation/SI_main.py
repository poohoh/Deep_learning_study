from training import *

img = cv2.imread('./Data/Training/t1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cb, cr = cv2.split(img)

# merged = cv2.merge([y, cb, cr])
# merged = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

training(y)