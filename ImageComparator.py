from abc import ABC, abstractmethod


class ImageComparator(ABC):

    @staticmethod
    def get_similarity_score(images, full_path_resolver_function=None):

        raise NotImplementedError

    @staticmethod
    def get_name():

        raise NotImplementedError

    @staticmethod
    def get_threshold():

        raise NotImplementedError
