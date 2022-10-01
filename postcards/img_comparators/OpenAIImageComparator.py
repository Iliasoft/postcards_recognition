#  https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb
import clip
import numpy as np
import torch
from PIL import Image

from .abstract_image_comparator import AbstractImageComparator


class OpenAIImageComparator(AbstractImageComparator):
    @staticmethod
    def get_similarity_score(images, full_path_resolver_function):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        preprocessed_images = []
        for image in images:
            try:
                preprocessed_images.append(
                    preprocess(
                        Image.open(full_path_resolver_function(image)).convert("RGB")
                    )
                )
            except OSError:
                print(
                    "skipping image due to disk error", full_path_resolver_function(image)
                )
                continue

        image_input = torch.tensor(np.stack(preprocessed_images)).cuda()
        with torch.no_grad():
            images_features = model.encode_image(image_input).float().cpu()

        images_features /= images_features.norm(dim=-1, keepdim=True)
        return images_features.cpu().numpy() @ images_features.cpu().numpy().T

    @staticmethod
    def get_name():

        return "OpenAI CLIP"

    @staticmethod
    def get_threshold():

        return 0.84
