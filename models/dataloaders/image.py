import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

img_1d = np.load('test.out.npy')

img_mat = img_1d.reshape(-224,224,3)

print(img_mat.shape)
print(img_mat.dtype)
tmp = img_mat.astype(int)

print(tmp.dtype)

img = cv.cvtColor(img_mat,cv.COLOR_RGB2BGR)

plt.imshow(img)
plt.show()
