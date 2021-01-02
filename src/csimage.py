from typing import List
import cv2
import numpy as np

class CSImage:
    """
    A class used to wrap an chess sheet image with useful functions to get specific details about the image
    """

    def __init__(self, img: np.ndarray, BINARY_THRESHOLD: int = 250) -> None:
        """
        :type img: numpy.ndarray of a standardized (fixed through some scan) chess sheet
        """

        self.img                    = img
        self.grayscale              = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.thresh              = cv2.threshold(self.grayscale, 
                                                    BINARY_THRESHOLD, 
                                                    2**8 - 1,
                                                    cv2.THRESH_BINARY_INV
                                                    )
        self.height, self.width, _  = img.shape
    
    def getContoursGreaterThan(self, minHeight: int, minWidth: int, DILATION_ITER: int = 14) -> List[np.ndarray]:
        """
        Returns contours within the given image over bounding rect minimum height and minimum width

        :type minHeight: int minimum height of contours within the image
        :type minWidth: int minimum width of contours within the image
        :type DILATION_ITER: int number of iterations to dilate the image, see cv2.dilate and BINARY_THRESHOLD
        :rtype List[np.ndarray]: returns a list of cv2 contours, can use cv2.boundingRect to get an approx rect that fit the contours
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        # Dilate image to get a rough outline where areas of interest are
        dilated = cv2.dilate(self.thresh, kernel, iterations = DILATION_ITER)

        # We won't be using the hierarchy of the contours, rather assuming that contours will include the two tables of moves in the game and just using the size of each contour to determine where each table is, and thus each move.
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # returns contours that fit the specific minHeight and minWidth
        return [ contour for contour in contours if (lambda c: c[2] > minWidth and c[3] > minHeight)(cv2.boundingRect(contour)) ]

if __name__ == '__main__':
    img = cv2.imread("Samples/sample_2.jpg")

    k = CSImage(img)

    def showImg(img):
        y, x = k.width, k.height
        n = cv2.resize(img, (y // 3, x // 3))
        cv2.imshow("test imge", n)
        cv2.waitKey(0)

    showImg(k.thresh)

    conts = k.getContoursGreaterThan(300, 300)

    cv2.drawContours(img, conts[0], -1, (0, 255, 0), 2)

    showImg(img)
