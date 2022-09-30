# https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
# https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
from __future__ import print_function
from __future__ import division
from tqdm import tqdm
import numpy as np

from ImageComparator import ImageComparator
import cv2 as cv
from BinaryComparisonMatrix import BinaryComparisionMatrix


class OpenCVHistogramBasedImageComparator(ImageComparator):

    @staticmethod
    def get_similarity_score(images, full_path_resolver_function):
        pairs = BinaryComparisionMatrix.get_pairs(np.arange(len(images)))
        comparison_results = np.empty((len(images), len(images)))
        images_histograms = []
        for image in images:

            src_base = cv.imread(full_path_resolver_function(image))
            hsv_base = cv.cvtColor(src_base, cv.COLOR_BGR2HSV)
            h_bins = 250
            s_bins = 260
            histSize = [h_bins, s_bins]
            # hue varies from 0 to 179, saturation from 0 to 255
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges # concat lists
            # Use the 0-th and 1-st channels
            channels = [0, 1]
            hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
            cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            images_histograms.append(hist_base)
            # base_base = cv.compareHist(hist_base, hist_base, compare_method)
            # base_half = cv.compareHist(hist_base, hist_half_down, compare_method)
        compare_method = 3# 0:1.01, 1:0 2:82, 3:0.4
        for pair in tqdm(pairs):

            compare_result = cv.compareHist(images_histograms[pair[0]], images_histograms[pair[1]], compare_method)
            #print(compare_result)
            comparison_results[pair[0], pair[1]] = compare_result
            comparison_results[pair[1], pair[0]] = compare_result

        return comparison_results
