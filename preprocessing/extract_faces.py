import numpy as np
import cv2 as cv
import os
from skimage.io import imread_collection

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

delta = 5
counter = 0

#returns an array of all images in a given folder and its subfolders
def load_images_from_folder(folder):
    images = []
    t = ()
    for subfolder in os.listdir(folder):
        test_img = cv.imread(os.path.join(folder, subfolder))
        print(os.path.join(folder, subfolder))
        if test_img is not None:
            images.append(subfolder, test_img)
        else:
            for filename in os.listdir(os.path.join(folder, subfolder)):
                print(os.path.join(folder, subfolder, filename))
                imgs = cv.imread(os.path.join(folder, subfolder, filename))
                if imgs is not None:
                    images.append(filename, imgs)
    return images

# folder where to search for pictures - no non folder/picture files or error
fp = "Inputfiles/imdb_crop"
cur_pad = os.path.normpath(fp)
image_tuples = load_images_from_folder(cur_pad)

for img_tuple in image_tuples:
    counter = 0
    img_name = img_tuple[0]
    img = img_tuple[1]

    faces = face_cascade.detectMultiScale(img, 1.8, 5)
    for (x,y,w,h) in faces:
        counter += 1
        roi_color = img[y:y+h, x:x+w]
        cip = img[y-delta:y+h+delta, x-delta:x+w+delta].copy()

        if counter > 0:
            facepath = "Outputfiles/" + img_name + str(counter) + ".jpg"
        else:
            facepath = "Outputfiles/" + img_name
        cur_path = os.path.normpath(facepath)
        print(cur_path)
        try:

            cip = cv.resize(cip, (224,224))
            b, g, r = cv.split(cip)
            b = b / 3
            g = g / 3
            r = r / 3
            meanRGB = cv.merge((b, g, r))
            cip = cip - meanRGB
            cv.imwrite(cur_path, cip)
        except:
            print("Resizing failed! !ssize.empty()")

cv.waitKey(0)
cv.destroyAllWindows()
