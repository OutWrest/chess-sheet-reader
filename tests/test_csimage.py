from csimage import CSImage
import cv2
import os

SAMPLES_PATH = "Samples/"

class Test_CSImage:
    def setup_class(self):
        fullSamples =  [ os.path.join(SAMPLES_PATH, file) for file in [ files for (_, _, files) in os.walk(SAMPLES_PATH) ][0] ]
        self.fullImgs = [ CSImage(cv2.imread(sample)) for sample in fullSamples ]
    
    def teardown_class(self):
        del self.fullImgs

    def test_cv2imageprocessing(self):
        for k in self.fullImgs:

            for prop in [k.height, k.width, k.grayscale, k.thresh]:
                assert prop is not None

    def test_tablecontourdection(self):
        for k in self.fullImgs:

            assert len(k.getContoursGreaterThan(k.height // 2, k.width // 4)) == 2