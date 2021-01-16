import numpy as np
from csimage import CSImage
import cv2
import math
import pytesseract

class CSReader:
    def __init__(self, img: np.ndarray) -> None:
        self.CSImg = CSImage(img)
        self.THICKNESS_C = 500000

        # Moves dict, move id to move str
        self.bestGuess = {} 

    def getWhiteMove(self, id: int):
        # TODO
        pass 

    def getBlackMove(self, id: int):
        # TODO
        pass

    def test(self):
        # get both table contours
        def showImg(img):
            y, x = self.CSImg.width, self.CSImg.height
            n = cv2.resize(img, (y // 4, x // 4))
            cv2.imshow("test imge", n)
            cv2.waitKey(0)

        tableContour = self.CSImg.tables[0]

        tableImg = self.CSImg.warptoContour(tableContour)

        width, height, *_ = tableImg.shape

        gray = cv2.cvtColor(tableImg, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 2**8 - 1, cv2.THRESH_BINARY_INV)

        Hmask = np.zeros((width, height, 3), np.uint8)
        Hmask[:] = (255, 255, 255)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//4, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations = 2)
        cnts, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts[:44]: # 44 h lines
            area = cv2.contourArea(c)
            if area > (height//45//15)*width:
                cv2.drawContours(Hmask, [c], -1, (0, 0, 0), (width*height)//self.THICKNESS_C)

        Vmask = np.zeros((width, height, 3), np.uint8)
        Vmask[:] = (255, 255, 255)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//4 ))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations = 2)
        cnts, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts[:2]: # 2 v lines (in the middle)
            area = cv2.contourArea(c)
            if area > (height//45//15)*width:
                cv2.drawContours(Vmask, [c], -1, (0, 0, 0), (width*height)//self.THICKNESS_C)

        pt_img = cv2.bitwise_or(Hmask, Vmask)

        pt_img = cv2.cvtColor(pt_img, cv2.COLOR_BGR2GRAY)

        cv2.drawContours(pt_img, cnts, -1, (0, 0, 255), 3)

        showImg(pt_img)



if __name__ == "__main__":
    img = cv2.imread("Samples/fullsample_1.jpg")

    k = CSReader(img)

    k.test()