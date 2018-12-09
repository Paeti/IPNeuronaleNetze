import os, time
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
delta = 5

#change directories here
path_to_watch = "tmp"
output_dir = "Output/"

before = dict ([(f, None) for f in os.listdir (path_to_watch)])
while 1:
    time.sleep (10)
    print("Alive")
    after = dict ([(f, None) for f in os.listdir (path_to_watch)])
    added = [f for f in after if not f in before]
    removed = [f for f in before if not f in after]
    if added:
        counter = 0
        for element in added:
            fp = path_to_watch + "/" + element
            cur_pad = os.path.normpath(fp)
            img = cv.imread(fp)
            faces = face_cascade.detectMultiScale(img, 1.8, 5)
            for (x, y, w, h) in faces:
                counter += 1
                roi_color = img[y:y + h, x:x + w]
                cip = img[y - delta:y + h + delta, x - delta:x + w + delta].copy()

                if counter > 0:
                    facepath = output_dir + added[0] + str(counter) + ".jpg"
                else:
                    facepath = output_dir + added[0]
                cur_path = os.path.normpath(facepath)
                print("Created: " + cur_path)
                try:

                    cip = cv.resize(cip, (224, 224))
                    b, g, r = cv.split(cip)
                    b = b / 3
                    g = g / 3
                    r = r / 3
                    meanRGB = cv.merge((b, g, r))
                    cip = cip - meanRGB
                    cv.imwrite(cur_path, cip)
                    print("Written: " + cur_path)
                except:
                    print("Resizing failed! !ssize.empty()")
            if os.path.exists(cur_pad):
                os.remove(cur_pad)

            else:
                print("File not found")
    if removed:
        print("Removed: ", ", ".join (removed))
    before = after
