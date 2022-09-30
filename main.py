from OpenCVHistogramBasedImageComparator import OpenCVHistogramBasedImageComparator
from GoogleLandmarksDS import GoogleLandmarksDS
from BinaryComparisonMatrix import BinaryComparisionMatrix
from OpenAIImageComparator import OpenAIImageComparator
from OpenCVTemplateBasedImageComparator import OpenCVTemplateBasedImageComparator
from SIFTImageComparator import SIFTImageComparator
import numpy as np
import pickle

if __name__ == '__main__':
    DS_CACHE = 'F:/MSAI/GoogleLandmarks2/data.pkl'
    try:
        pkl_file = open(DS_CACHE, 'rb')
        myDS = pickle.load(pkl_file)
        print("Google Landmarks DS: Loaded cached data from disk")

    except FileNotFoundError as error:
        myDS = GoogleLandmarksDS()

        output = open(DS_CACHE, 'wb')
        pickle.dump(myDS, output, -1)
        output.close()
        print("Google Landmarks DS: Cached data saved to disk")


    # how many images from dataset we want to compare with each other?
    max_images_to_be_compared = 500
    images_to_compare = []
    actually_linked_images = myDS.get_images_linked_by_landmark()

    for picture1, picture2 in actually_linked_images:

        if picture1 not in images_to_compare:
            images_to_compare.append(picture1)

        if picture2 not in images_to_compare:
            images_to_compare.append(picture2)

    images_comparators = [OpenAIImageComparator(), SIFTImageComparator()]
    possible_image_pairs = BinaryComparisionMatrix.get_pairs(np.arange(len(images_to_compare)))

    print(f"All images:{len(myDS.images)}, Images with landmarks:{len(myDS.images2landmarks)}, Unique images with landmarks: {len(images_to_compare)}, Landmarks:{len(myDS.landmarks2images)}, Possible pairs:{len(possible_image_pairs)}")

    for comparator in images_comparators:
        similarity_predictions = comparator.get_similarity_score(images_to_compare, myDS.get_image_full_path)
        possible_image_pairs = BinaryComparisionMatrix.get_pairs(np.arange(similarity_predictions.shape[0]))

        tp, tn, fp, fn = 0, 0, 0, 0
        # check if predicted links are actually correct
        for possible_image_pair in possible_image_pairs:

            ground_truth = myDS.is_linked_by_same_landmark(
                images_to_compare[possible_image_pair[0]],
                images_to_compare[possible_image_pair[1]]
            )

            prediction = True if similarity_predictions[possible_image_pair[0], possible_image_pair[1]] >= comparator.get_threshold() else False

            if ground_truth is True and prediction is True:
                tp += 1
            elif ground_truth is True and prediction is False:
                fn += 1
            elif ground_truth is False and prediction is True:
                fp += 1
            elif ground_truth is False and prediction is False:
                tn += 1

        print(f"Prediction results for {comparator.get_name()}, Precision: {tp / (tp + fp):.3f}, Recall: {tp / (tp + fn):.3f}, True links: {len(actually_linked_images)}, [TP:{tp}, FN:{fn}, FP:{fp}, TN:{tn}], TP-FN-FP = {tp - fn- fp}")
