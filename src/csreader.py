import numpy as np
from csimage import CSImage
import cv2
from sympy import Point, Line, Segment
import math
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"

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

    def ocr(self, img):
        data = pytesseract.image_to_string(img, lang='eng', config="--psm 13 --oem 3 -c tessedit_char_whitelist=Move")
        print(data.encode())

    def test(self, epsilon: float = 0.009):
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
        hcnts, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #for c in cnts[:44]: # 44 h lines
        #    area = cv2.contourArea(c)
        #    if area > (height//45//15)*width:
        #        cv2.drawContours(Hmask, [c], -1, (0, 0, 0), (width*height)//self.THICKNESS_C)

        Vmask = np.zeros((width, height, 3), np.uint8)
        Vmask[:] = (255, 255, 255)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//4 ))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations = 2)
        vcnts, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #for c in cnts[:2]: # 2 v lines (in the middle)
        #    area = cv2.contourArea(c)
        #    if area > (height//45//15)*width:
        #        cv2.drawContours(Vmask, [c], -1, (0, 0, 0), (width*height)//self.THICKNESS_C)

        pt_img = cv2.bitwise_or(Hmask, Vmask)

        #pt_img = cv2.cvtColor(pt_img, cv2.COLOR_BGR2GRAY)

        pt = []
        
        #for vcnt in vcnts:
        #    varea = cv2.contourArea(vcnt)
        #    if varea > (height//45//15)*width:
        #        for hcnt in hcnts[:44]:
        #            harea = cv2.contourArea(hcnt)
        #            if harea > (height//45//15)*width:

        hcnts.reverse() 
        vcnts = sorted(vcnts, key = lambda x: (sum(cv2.boundingRect(x))))

        for hcnt in hcnts[:44]:
            harea = cv2.contourArea(hcnt)
            if harea > (height//45//15)*width:
                for vcnt in vcnts[:4]:
                    varea = cv2.contourArea(vcnt)
                    if varea > (height//45//15)*width:
                        vapprox = cv2.approxPolyDP(vcnt, epsilon * cv2.arcLength(vcnt, True), True)
                        happrox = cv2.approxPolyDP(hcnt, epsilon * cv2.arcLength(hcnt, True), True)

                        assert len(vapprox) == 2 # Line segment
                        assert len(happrox) == 2 # Line segment

                        pt1, pt2 = sorted(vapprox, key = lambda x: sum(sum(x)))
                        vline = Line(Point(*pt1, evaluate=False), Point(*pt2, evaluate=False))

                        pt1, pt2 = sorted(happrox, key = lambda x: sum(sum(x)))
                        hline = Line(Point(*pt1, evaluate=False), Point(*pt2, evaluate=False))

                        intersect = vline.intersection(hline)

                        assert len(intersect) == 1 # Only intersects 1 once

                        pt.append(tuple(map(round, intersect[0].coordinates)))

                        # cv2.circle(tableImg, tuple(map(round, intersect[0].coordinates)), 5, (255, 0, 0), 15)

                        # showImg(tableImg)

        
        print(len(pt))
        showImg(tableImg)

        # for every 4 pt on grid, warpaffix for later use. current is fine

        for i in range(0, 5 - 4, 4):
            l, lm, rm, r, bl, blm, brm, br = pt[i:i+8]

            # move
            w, h = blm[0]-l[0], blm[1]-l[1]
            x, y = l
            move = tableImg[y:y+h, x:x+w]

            showImg(move) 
            self.ocr(move)        

            print(f"[{i}]: ", l, lm, rm, r, bl, blm, brm, br)



if __name__ == "__main__":
    img = cv2.imread("Samples/fullsample_1.jpg")

    k = CSReader(img)

    k.test()