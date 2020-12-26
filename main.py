import numpy as np
from os.path import join
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'


# Get image path
path, file_name = 'Samples', 'sample_2.jpg'

# Read image
img = cv2.imread(join(path, file_name))

RS_FACTOR = 3

def showImg(img):
    x, y = img.shape[0], img.shape[1]
    n = cv2.resize(img, (y * RS_FACTOR, x * RS_FACTOR))
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

tables = []
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    if h>300 and w>300:
        tables.append([x,y,w,h])

x, y, w, h = tables[0]

right_table = img[y:y+h, x:x+w]
rt_thresh = thresh[y:y+h, x:x+w]

contours, hierarchy = cv2.findContours(rt_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

bound = max([(cv2.contourArea(cnt), cnt) for cnt in contours], key=lambda x: x[0])[1]

#x, y, w, h = cv2.boundingRect(bound)

#right_table = right_table[y:y+h, x:x+w]
#rt_thresh = rt_thresh[y:y+h, x:x+w]

#print(cv2.minAreaRect(bound))

cv2.imwrite('test.jpg', right_table)

approx = cv2.approxPolyDP(bound, 0.009 * cv2.arcLength(bound, True), True) 
# cv2.drawContours(right_table, [approx], -1, (0, 0, 255), 5)

#showImg(right_table)

(tr_x, tr_y, tl_x, tl_y, bl_x, bl_y, br_x, br_y) = approx.ravel()

x, y, _ = right_table.shape

pts1 = np.float32([[tr_x, tr_y], [tl_x, tl_y], [bl_x, bl_y], [br_x, br_y]])
pts2 = np.float32([[y, 0], [0, 0], [0, x], [y, x]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(right_table, M, (y, x))

#showImg(dst)

# Test tesseract

img_pytes_test = cv2.imread("b.jpg")
# grayscale
gray = cv2.cvtColor(img_pytes_test,cv2.COLOR_BGR2GRAY) 

# threshold
_,thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV) 

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
bound = max([(cv2.contourArea(cnt), cnt) for cnt in contours], key=lambda x: x[0])[1]
rect = cv2.minAreaRect(bound)
box = cv2.boxPoints(rect)
box = np.int0(box)

print(box)

# cv2.drawContours(img_pytes_test, [box], -1, (0, 0, 255), 2)

[br_x, br_y], [bl_x, bl_y], [tl_x, tl_y], [tr_x, tr_y] = box

showImg(img_pytes_test[tl_y:br_y, tl_x:br_x])

data = pytesseract.image_to_data(img_pytes_test[tl_y:br_y, tl_x:br_x], output_type=pytesseract.Output.DICT, config="--psm 7")

print(data['text'])



