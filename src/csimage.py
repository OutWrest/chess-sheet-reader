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
    
    def getContoursGreaterThan(self, minHeight: int, minWidth: int) -> List[np.ndarray]:
        """
        TODO
        """
        pass





if __name__ == '__main__':
    img = cv2.imread("Samples/sample_2.jpg")

    k = CSImage(img)

    def showImg(img):
        y, x = k.width, k.height
        n = cv2.resize(img, (y // 3, x // 3))
        cv2.imshow("test imge", n)
        cv2.waitKey(0)

    showImg(k.thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(k.thresh,kernel,iterations = 14) 
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

    #print(type(contours[0]))
