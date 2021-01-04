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
        self.height, self.width, *_  = img.shape
    
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
        # dilated = cv2.dilate(self.thresh, kernel, iterations = DILATION_ITER)

        # We won't be using the hierarchy of the contours, rather assuming that contours will include the two tables of moves in the game and just using the size of each contour to determine where each table is, and thus each move.
        contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.drawContours(self.img, contours, -1, (255, 0, 0), 5)

        # returns contours that fit the specific minHeight and minWidth
        return [ contour for contour in contours if (lambda c: c[2] > minWidth and c[3] > minHeight)(cv2.boundingRect(contour)) ]

    
    def getBetterApprox(self, cnt, epsilon: float = 0.009):
        # Get better approximate of contour, lower dilation + polyDP

        # Get a bounding box of the contour that needs a better approx
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Using the thresh img, get the contour without dilation iterations 
        cnts, _ = cv2.findContours(self.thresh[y:y+h, x:x+w], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bound = max([ (cv2.contourArea(cnt), cnt) for cnt in cnts ], key = lambda x: x[0])[1]

        approx = cv2.approxPolyDP(bound, epsilon * cv2.arcLength(bound, True), True)

        tl, tr, bl, br = sorted(approx, key = lambda x: sum(sum(x)))

        newIm = self.img[y:y+h, x:x+w]

        rect = np.float32([[*tr], [*tl], [*bl], [*br]])
        rect_mapped = np.float32([[h, 0], [0, 0], [0, w], [h, w]])

        M = cv2.getPerspectiveTransform(rect, rect_mapped)
        dst = cv2.warpPerspective(newIm, M, (h, w))

        return dst

    def cropApproxAndWarp(self, approx, ) -> np.ndarray:
        pass 
    '''
        # Transform image to fix contour box
        tl, tr, bl, br = sorted(approx, key = lambda x: sum(sum(x)))

        newIm = self.img[y:y+h, x:x+w]

        rect = np.float32([[*tr], [*tl], [*bl], [*br]])
        rect_mapped = np.float32([[h, 0], [0, 0], [0, w], [h, w]])

        M = cv2.getPerspectiveTransform(rect, rect_mapped)
        dst = cv2.warpPerspective(newIm, M, (h, w))

        return dst 
        '''

    @property
    def tables(self):
        # Assumes a at least a 1:2 and 1:4 ratio, [left contour, right contour]
        t1, t2 = self.getContoursGreaterThan(self.height // 2, self.width // 4)
        return sorted((t1, t2), key=lambda x: x[0][0][0])
    
    def splitTablesByInferance(self):
        # Split image by knowing how big the table is after getting a better approx
        pass

if __name__ == '__main__':
    img = cv2.imread("Samples/fullsample_1.jpg")

    k = CSImage(img)

    def showImg(img):
        y, x = k.width, k.height
        n = cv2.resize(img, (y // 4, x // 4))
        cv2.imshow("test imge", n)
        cv2.waitKey(0)

    #(x, y, w, h) = cv2.boundingRect(k.tables[0])

    #cnts, _ = cv2.findContours(k.thresh[y:y+h, x:x+w], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #bound = max([ (cv2.contourArea(cnt), cnt) for cnt in cnts ], key = lambda x: x[0])[1]

    c = k.getBetterApprox(k.tables[1])

    #cv2.drawContours(k.img, [c], -1, (0, 0, 255), 3)

    showImg(c)
