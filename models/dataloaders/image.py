import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

img_1d = np.load('test2.out.npy')

img_mat = img_1d.reshape(-224,224,3)

print(img_mat.shape)
print(img_mat.dtype)
tmp = img_mat.astype(int)

print(tmp.dtype)
print(tmp.shape)

#print(tmp[:,:,0])
#print(tmp[:,:,1])
#print(tmp[:,:,2])


print(img_mat[:,:,0])
print(img_mat[:,:,1])
print(img_mat[:,:,2])


#img = cv.cvtColor(tmp,cv.COLOR_BGR2RGB)

#cv.imwrite('./farbschema2.jpg',img_mat)

#(B, G, R) = cv.split(img_mat)
#merged = cv.merge([R,G,B])

#cv.imshow("image",img_mat)

plt.imshow(img_mat)
plt.show()


reloaded_img = cv.imread('farbschema2.jpg')
print(reloaded_img.shape)

print("____________________________________________________________")
print(reloaded_img[:,:0])
print(reloaded_img[:,:,1])
print(reloaded_img[:,:,2])

plt.imshow(reloaded_img)
plt.show()
