import numpy as np
from os.path import join
import cv2

# Get image path
path, file_name = 'Samples', 'sample_2.jpg'

# Read image
img = cv2.imread(join(path, file_name))

def showImg(img):
    x, y = img.shape[0], img.shape[1]
    n = cv2.resize(img, (y // 4, x // 4))
    cv2.imshow(file_name, n)
    cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

# threshold
_,thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV) 

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

# dilate
dilated = cv2.dilate(thresh,kernel,iterations = 14) 

# get contours
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>300 and w>300:
        #continue
        pass

    # discard areas that are too small
    if h<40 or w<40:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

showImg(img)







