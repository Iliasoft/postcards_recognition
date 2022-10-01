class BinaryComparisionMatrix:
    @staticmethod
    def get_pairs(images_to_compare):
        list_of_images = list(images_to_compare)
        pairs = []
        for outer_idx in range(len(list_of_images)):
            for inner_idx in range(outer_idx + 1, len(list_of_images)):
                assert list_of_images[outer_idx] != list_of_images[inner_idx]
                pairs.append((list_of_images[outer_idx], list_of_images[inner_idx]))

        return pairs
