import numpy as np
import os
from scipy.spatial import distance

from skimage.metrics import structural_similarity as ssim
import cv2

import pickle



def match_user_histogram(user_hist, histogram_dir="./colour_histograms", top_n=5):
    """
    Compares a user histogram to all histograms in the database using Euclidean distance.

    :param user_hist: NumPy array of the user's color histogram
    :param histogram_dir: Path to the saved database histograms (.npy files)
    :param top_n: Number of top matches to return
    :return: List of tuples (image_name, distance) sorted by closest match
    """
    anime_similarity = {}

    for file in os.listdir(histogram_dir):
        if file.endswith(".npy"):
            # hist_path = os.path.join(histogram_dir, file)
            # db_hist = np.load(hist_path)
            # dist = distance.euclidean(user_hist, db_hist)

            hist = np.load(os.path.join(histogram_dir, file))
            dist = distance.euclidean(user_hist, hist)

            similarity = 1 / (1 + dist)
            similarity_percentage = similarity * 100

            image_name = os.path.splitext(file)[0]
            anime_name = image_name.split("__")[0]  # extract anime name from filename

            # Keep only the highest similarity per anime
            if anime_name not in anime_similarity or dist < anime_similarity[anime_name]:
                anime_similarity[anime_name] = dist

    # Convert dict to list and sort by similarity
    sorted_results = sorted(anime_similarity.items(), key=lambda x: x[1])
    return sorted_results[:top_n]


