from csimage import CSImage
import cv2

img = cv2.imread("temp/Samples/sample_2.jpg")
k = CSImage(img)

# Assumes no other contour is greater than 300px except tables
# TODO: remove aassumation and standardize a ratio

def test_tablecontourdection():
    assert len(k.getContoursGreaterThan(300, 300)) == 2