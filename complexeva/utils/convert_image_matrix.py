#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes


def image_to_adjacency_matrix(image_path, threshold=128, hole_area_threshold_ratio=0.05, save_as=None):
    """
    Convert an image to a binary adjacency matrix using scikit-image preprocessing.
    - save_as: The path to save the processed binary image.
    - threshold: The threshold value for binarization. max value is 255.
    - hole_area_threshold_ratio: The ratio of the hole area to the image area to be removed (ratio to the diagnal length).
    """
    image = imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[..., :3]
        image = rgb2gray(image)

    threshold = threshold_otsu(image)
    binary_image = image > threshold

    # ---- some clean up ----
    # binary_image = remove_small_objects(binary_image, min_size=64)
    area_threshold = hole_area_threshold_ratio * np.hypot(*binary_image.shape)
    binary_image = remove_small_holes(binary_image, area_threshold=area_threshold)
    binary_matrix = binary_image.astype(np.uint8)

    if save_as:
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(imread(image_path), interpolation="nearest")
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(binary_matrix, cmap="binary", interpolation="nearest")
        plt.title("Processed Binary Image")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.close()
    return binary_matrix
