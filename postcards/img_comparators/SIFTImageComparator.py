# https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
import cv2 as cv
import numpy as np
from tqdm import tqdm

from postcards.BinaryComparisonMatrix import BinaryComparisionMatrix

from .abstract_image_comparator import AbstractImageComparator


class SIFTImageComparator(AbstractImageComparator):
    @staticmethod
    def get_similarity_score(images, full_path_resolver_function):

        threshold = 0.35
        # print("TH:", threshold)
        pairs = BinaryComparisionMatrix.get_pairs(np.arange(len(images)))
        comparison_results = np.empty((len(images), len(images)))
        images_descriptors = []
        for image in images:
            image = cv.imread(full_path_resolver_function(image))
            ####################################################

            # SIFT option
            # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            kp, des = sift.detectAndCompute(image, None)

            """
            # SURF Option
            surf = cv.xfeatures2d.SURF_create(400)
            surf.setUpright(True)
            surf.setExtended(True)
            kp, des = surf.detectAndCompute(image, None)
            """
            ####################################################
            images_descriptors.append(des)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        pairs = BinaryComparisionMatrix.get_pairs(np.arange(len(images)))
        for pair in tqdm(pairs):

            matches = flann.knnMatch(
                images_descriptors[pair[0]], images_descriptors[pair[1]], k=2
            )
            positive_matches = 0
            for m, n in matches:
                positive_matches += 1 if m.distance < threshold * n.distance else 0

            comparison_results[pair[0], pair[1]] = positive_matches
            comparison_results[pair[1], pair[0]] = comparison_results[pair[0], pair[1]]

        # print(np.min(comparison_results), np.max(comparison_results))
        return comparison_results

    @staticmethod
    def get_name():

        return "OpenCV SIFT/FLANN"

    @staticmethod
    def get_threshold():

        return 3
