import cv2
import numpy as np
import os
from sklearn.cluster import KMeans


def extract_dominant_colour(image, k=3):
    """
    Extract the dominant colour of an image using k-means clustering on its pixels.

    Parameters:
        image (numpy.ndarray): The image in RGB format.
        k (int): The number of clusters to form for extracting dominant colour.

    Returns:
        dominant_colour (tuple): The dominant colour in RGB (as integers).
    """
    # Resize image to speed up processing
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    # Reshape image data to a 2D array of pixels and convert to float32
    pixels = resized.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define criteria and apply k-means clustering
    # (stop criteria: either 10 iterations or epsilon = 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count the number of pixels assigned to each cluster
    counts = np.bincount(labels.flatten())

    # The dominant colour is the cluster center with the largest count
    dominant = centers[np.argmax(counts)]

    # Convert colour values to integers
    dominant_colour = tuple(map(int, dominant))
    return dominant_colour


def group_images_by_dominant_colour(image_paths, final_clusters=20, k_per_image=3):
    """
    Groups images by their dominant colour.

    Parameters:
        image_paths (list): List of image file paths.
        final_clusters (int): Number of clusters to group images by dominant colour.
        k_per_image (int): Number of clusters used to extract the dominant colour from each image.

    Returns:
        dict: Dictionary mapping cluster label to a list of image paths.
    """
    # List to store dominant colours and corresponding image paths
    dominant_colours = []
    valid_image_paths = []

    for path in image_paths:
        # Read image in BGR and convert to RGB
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image at {path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dominant = extract_dominant_colour(image_rgb, k=k_per_image)
        dominant_colours.append(dominant)
        valid_image_paths.append(path)

    # Convert list of dominant colours to numpy array for clustering
    dominant_colours = np.array(dominant_colours)

    # Cluster the dominant colours into final_clusters groups
    kmeans_final = KMeans(n_clusters=final_clusters, random_state=42)
    labels = kmeans_final.fit_predict(dominant_colours)

    # Build a dictionary mapping cluster label to image paths
    clusters = {i: [] for i in range(final_clusters)}
    for label, path in zip(labels, valid_image_paths):
        clusters[label].append(path)

    return clusters


def save_clusters(clusters, output_folder="./clusters"):
    """
    Saves images grouped by their cluster label into separate subfolders.

    Parameters:
        clusters (dict): Dictionary mapping cluster labels to lists of image paths.
        output_folder (str): Directory where clusters will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for cluster_label, image_paths in clusters.items():
        cluster_dir = os.path.join(output_folder, f"cluster_{cluster_label}")
        os.makedirs(cluster_dir, exist_ok=True)

        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image at {image_path}")
                continue

            filename = os.path.basename(image_path)
            dest_path = os.path.join(cluster_dir, filename)
            cv2.imwrite(dest_path, image)

    print(f"Clusters have been saved to the folder: {output_folder}")




