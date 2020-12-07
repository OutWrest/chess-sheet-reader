import numpy as np
import cv2

img = cv2.imread('africa.png')
print(type(img))
cv2.imshow('africa', img)
k = cv2.waitKey(0)

# [
#   [ [255, 255, 255], ... ],
#   [ [R, G, B], ... ],
#   ...
# ]