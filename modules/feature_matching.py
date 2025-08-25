from feature_extraction import extract_canny_edges
from skimage.metrics import structural_similarity as ssim

import os
import cv2
import numpy as np
from scipy.spatial import distance
from feature_extraction import extract_sift_descriptors



def match_user_features(user_hist, user_img, histogram_dir="./colour_histograms", preprocessed_dir="./processed_images", sift_dir="./sift_descriptors", top_n=5):
    anime_scores = {}

    # Extract SIFT descriptors from the user's input
    user_sift = extract_sift_descriptors(user_img)


    for file in os.listdir(histogram_dir):
        if file.endswith(".npy"):
            image_name = os.path.splitext(file)[0]
            anime_name = image_name.split("__")[0]

            # Load histogram
            hist_path = os.path.join(histogram_dir, file)
            db_hist = np.load(hist_path)
            colour_dist = distance.euclidean(user_hist, db_hist)
            colour_similarity = 1 / (1 + colour_dist)  # Higher is better

            # Load SIFT descriptors
            sift_path = os.path.join(sift_dir, file)
            if os.path.exists(sift_path):
                db_sift = np.load(sift_path, allow_pickle=True)
            else:
                db_sift = None


            # Match SIFT features
            sift_matches = match_sift_features(user_sift, db_sift)
            sift_similarity = sift_matches / (sift_matches + 20)

            # Combine scores
            combined_score = ((colour_similarity * 0.3) + (sift_similarity * 0.7)) * 100

            # Keep the highest combined score per anime
            if anime_name not in anime_scores or combined_score > anime_scores[anime_name]:
                anime_scores[anime_name] = combined_score

    # Sort by combined score
    sorted_results = sorted(anime_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:top_n]







def match_sift_features(query_descriptors, db_descriptors):
    """
    Matches SIFT descriptors between query and database image.

    :param query_descriptors: SIFT descriptors from query image
    :param db_descriptors: SIFT descriptors from database image
    :return: Number of good matches (higher is better)
    """
    if query_descriptors is None or db_descriptors is None:
        return 0

    # FLANN parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(query_descriptors, db_descriptors, k=2)
    except cv2.error:
        return 0  # In case of no matches possible

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(good_matches)


