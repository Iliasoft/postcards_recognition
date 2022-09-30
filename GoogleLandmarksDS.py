# This class implements interfaces to Google Landmarks DS
from torch.utils.data import Dataset
import csv
import os.path
from tqdm import tqdm
from PIL import Image

DATA_PATH = 'F:/MSAI/GoogleLandmarks2/'


class GoogleLandmarksDS(Dataset):

    def __init__(self, root_dir=DATA_PATH, transformer=None):

        self.root_dir = root_dir
        self.transformer = transformer
        self.images = []
        self.landmarks2images = dict()
        self.images2landmarks = dict()

        with open(root_dir + 'recognition_solution_v2.1.csv') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            header_extracted = False
            for row in tqdm(csv_reader, desc="Parsing dataset images"):
                if not header_extracted:
                    header_extracted = True
                    continue

                file_id = row[0]
                file_full_name = self.get_image_full_path(file_id)
                if not os.path.exists(file_full_name):
                    continue

                if file_id not in self.images:
                    self.images.append(file_id)

                if len(row[1]):
                    self.images2landmarks[file_id] = row[1].split()

                    for landmark in self.images2landmarks[file_id]:
                        if landmark not in self.landmarks2images:
                            s = set()
                            s.add(row[0])
                            self.landmarks2images[landmark] = s
                        else:
                            self.landmarks2images[landmark].add(row[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_id = self.images[index]
        return img_id, self.images2landmarks[img_id] if img_id in self.images2landmarks else None

    def get_image_full_path(self, id):
        return self.root_dir + id[0] + '/' + id[1] + '/' + id[2] + '/' + id + ".jpg"

    def is_linked_by_same_landmark(self, id1, id2):
        # true if both images are connected to a same landmark
        same_landmark = False

        assert id1 != id2
        if id1 in self.images2landmarks and id2 in self.images2landmarks:
            for landmark in self.images2landmarks[id1]:
                if landmark in self.images2landmarks[id2]:
                    same_landmark = True
                    break
        return same_landmark

    def get_images_linked_by_landmark(self):
        # true if both images are connected to a same landmark
        links = []
        for landmark in self.landmarks2images.keys():
            imgs_of_landmark = list(self.landmarks2images[landmark])

            for outer_idx in range(len(imgs_of_landmark)):
                for inner_idx in range(outer_idx + 1, len(imgs_of_landmark)):
                    # assert imgs_of_landmark[outer_idx] != imgs_of_landmark[inner_idx]
                    if (imgs_of_landmark[outer_idx], imgs_of_landmark[inner_idx]) not in links:
                        links.append(
                            (
                                imgs_of_landmark[outer_idx],
                                imgs_of_landmark[inner_idx]
                            )
                        )

        return links

    def get_image_idx_by_id(self, id):
        return self.images.index(id)
