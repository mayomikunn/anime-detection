from modules.feature_extraction import extract_colour_histogram
from user_input import get_user_image
from database import load_database_images, get_all_image_paths
from preprocessing import preprocess_image
from clustering import group_images_by_dominant_color, save_clusters
from feature_extraction import extract_preprocessed_images
from feature_matching import match_user_histogram

import os
import cv2










def main():
    # Preprocess the User Image
    user_image_path = get_user_image()

    # Preprocess the user image
    processed_image = preprocess_image(user_image_path)

    if processed_image is not None:
        print("User image preprocessed successfully!")


    if extract_preprocessed_images is not None:
        print("Extracted user colour histograms successfully!")




    # Check if preprocessed data file exists before re-processing everything
    preprocessed_file = "processed_images"
    if not os.path.exists(preprocessed_file):
        print("Preprocessed data not found. Running preprocessing...")
        load_database_images()  # Adjust parameters as needed
    else:
        print(f"Using existing preprocessed data from {preprocessed_file}")


    # Check if colour histogram file exists before running extraction
    colour_histogram_file = "colour_histograms"
    if not os.path.exists(colour_histogram_file):
        print("Colour histogram data not found. Running colour histogram...")
        extract_preprocessed_images("./processed_images")
    else:
        print(f"Using existing colour histogram data from {colour_histogram_file}")

    # Match against database
    if processed_image is not None:
        user_histogram = extract_colour_histogram(processed_image)
        top_matches = match_user_histogram(user_histogram, histogram_dir="./colour_histograms", top_n=5)

        print("\nTop matches based on colour similarity:")
        for idx, (anime_name, dist) in enumerate(top_matches, 1):
            print(f"{idx}. {anime_name} - Distance: {dist:.4f}")


        # print("\nTop matches based on colour similarity:")
        # for idx, (anime_name, similarity) in enumerate(top_matches, 1):
        #     print(f"{idx}. {anime_name} - Similarity: {similarity:.2f}%")

    # Cluster Database Images by Dominant Colour (Before Preprocessing)
    # base_dir = "C:/Users/mayom/Downloads/archive/data/anime_images"
    # print("Collecting image paths for clustering...")
    # image_paths = get_all_image_paths(base_dir)
    #
    # if image_paths:
    #     print("Running clustering on database images...")
    #     clusters = group_images_by_dominant_color(image_paths, final_clusters=20, k_per_image=3)
    #     print("Clustering completed. Cluster summary:")
    #     # Save clusters to folder for visual inspection
    #     save_clusters(clusters, output_folder="clusters")
    #     for cluster_id, paths in clusters.items():
    #         print(f"  Cluster {cluster_id}: {len(paths)} images")
    # else:
    #     print("No images found for clustering.")




if __name__ == "__main__":
    main()