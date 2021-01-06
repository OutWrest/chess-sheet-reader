import numpy as np
from csimage import CSImage
import cv2
import pytesseract

class CSReader:
    def __init__(self, img: np.ndarray) -> None:
        self.CSImg = CSImage(img)

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

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//2,1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(tableImg, [c], -1, (36,255,12), 2)

        showImg(tableImg)


if __name__ == "__main__":
    img = cv2.imread("Samples/fullsample_1.jpg")

    k = CSReader(img)

    k.test()