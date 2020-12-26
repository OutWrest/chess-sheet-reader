import numpy as np
from os.path import join
import cv2

# Get image path
path, file_name = 'Samples', 'sample_2.jpg'

# Read image
img = cv2.imread(join(path, file_name))

VIEW_IMAGE_FACTOR = 4
THRESHOLD_VALUE = 50
THRESHOLD_ITER = 5
MIN_XLINE = (img.shape[0] * 700)//1024
MIN_YLINE = 20
MIN_XGAP = (img.shape[0] * 20)//792
MIN_YGAP = 3
HLP_THRES = 80
MAX_PIXEL_ROT = 300

# Convert image around

def getThresh(img_t):
    img_grayscale = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_grayscale)
    ret, img_thresh = cv2.threshold(img_invert, THRESHOLD_VALUE, 2**8 - 1, cv2.THRESH_BINARY)
    return img_thresh

img_thresh = getThresh(img)

# Adaptive testing
# th3 = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
# cv2.imshow(file_name, th3)
# k = cv2.waitKey(0)

def showImg(img):
    #print(img.shape)
    x, y, _ = img.shape
    n = cv2.resize(img, (y // VIEW_IMAGE_FACTOR, x // VIEW_IMAGE_FACTOR))
    cv2.imshow(file_name, n)
    cv2.waitKey(0)

# Display image
#showImg(img_thresh)

# [
#   [ [255, 255, 255], ... ],
#   [ [R, G, B], ... ],
#   ...
# ]

'''

kernel_length = np.array(img).shape[1]//80

verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img_temp1 = cv2.erode(img_thresh, verticle_kernel, iterations=THRESHOLD_ITER)
#showImg(img_temp1)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=THRESHOLD_ITER)
#showImg(verticle_lines_img)


img_temp2 = cv2.erode(img_thresh, hori_kernel, iterations=THRESHOLD_ITER)
#showImg(img_temp2)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=THRESHOLD_ITER)
#showImg(horizontal_lines_img)

#showImg(np.add(horizontal_lines_img, verticle_lines_img))


alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    if (w > 250 and w < 300 and h < 25 and h > 10) and w > 2*h:
        showImg(img[y:y+h, x:x+w])

showImg(cv2.drawContours(img, contours, -1, (0,255,0), 3))
showImg(img)

'''

def test(img_t):
    edges = cv2.Canny(getThresh(img_t), 50, 250)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HLP_THRES, minLineLength=MIN_XLINE, maxLineGap=400)

    cv2.imshow(file_name, edges)
    cv2.waitKey(0)

    vertical_lines = sorted([l[0] for l in lines if abs(l[0][0]-l[0][2]) < MAX_PIXEL_ROT], key=lambda x:x[0])

    prevX = 0
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        # Min Gap
        if x1-prevX > MIN_XGAP: 
            prevX = x1
            cv2.line(img_t, (x1, y1), (x2, y2), (255, 0, 0), 3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HLP_THRES, minLineLength=MIN_YLINE, maxLineGap=100*4)

    horizontal_lines = sorted([l[0] for l in lines if abs(l[0][1]-l[0][3]) < MAX_PIXEL_ROT], key=lambda x:x[1])

    yDif = []

    prevY = 0
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        # Min Gap
        if y1-prevY > MIN_YGAP:
            cv2.line(img_t, (x1, y1), (x2, y2), (255, 0, 0), 3)
            yDif.append(abs(prevY-y1))
            prevY = y1

    print(max(set(yDif), key=yDif.count))

test(img)
#test(img_right)

#showImg(img_left)
showImg(img)
