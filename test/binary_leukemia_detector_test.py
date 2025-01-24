import unittest
import cv2
import numpy as np

from binary_leukemia_detector import ALLConfig, ImageFeatureExtractor, EnsembleALLDetector


class BinaryLeukemiaDetectorTest(unittest.TestCase):
    def setUp(self):
        self.path_image = "/dataset/C-NMC_Leukemia/training_data/fold_0/all/UID_1_1_1_all.bmp"
        self.config = ALLConfig()
        self.image_feature_extractor = ImageFeatureExtractor(self.config)
        self.ensemble_all_detector = EnsembleALLDetector(self.config)


    def test_init(self):

        self.assertEqual(self.config.test_size, 0.2)
        self.assertIsNotNone(self.config.rf_params)
        self.assertIsNotNone(self.config.etc_params)
        self.assertIsNotNone(self.config.svm_params)
        self.assertIsNotNone(self.config.lr_params)

    def test_extract_image_features(self):
        image : np.array =  cv2.imread(self.path_image)
        print(image.shape)
        np.savetxt("test.txt", image.reshape(-1, image.shape[-1]))
        self.assertIsInstance(image, np.ndarray)
        result : np.array = extract_features(image)
        self.assertIsInstance(result, np.ndarray)
        print(f"result : {result.shape}")

    def test_train(self) :
        self.ensemble_all_detector.train()

if __name__ == '__main__':
    unittest.main()