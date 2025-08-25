from modules.feature_extraction import extract_colour_histogram, extract_preprocessed_images, extract_canny_edges, \
    extract_sift_descriptors, display_image_and_sift_keypoints
from user_input import get_user_image
from database import load_database_images, get_all_image_paths
from preprocessing import preprocess_image
from feature_matching import match_user_features
from visualise_results import display_anime_results

import os


# from clustering import group_images_by_dominant_colour, save_clusters




def main():
    # Step 1: Get User Image
    user_image_path = get_user_image()

    # Step 2: Preprocess User Image
    processed_image = preprocess_image(user_image_path)
    if processed_image is not None:
        print("User image preprocessed successfully!")
        display_image_and_sift_keypoints(user_image_path)
    else:
        print("Error: User image preprocessing failed.")
        return

    # Step 3: Ensure Preprocessed Database Images Exist
    preprocessed_dir = "processed_images"
    if not os.path.exists(preprocessed_dir):
        print("Preprocessed images not found. Processing database images...")
        load_database_images()
    else:
        print(f"Using existing preprocessed images from '{preprocessed_dir}'")

    # Step 4: Ensure Colour Histograms and SIFT Descriptors Exist
    colour_histogram_dir = "colour_histograms"
    sift_descriptor_dir = "sift_descriptors"

    if not os.path.exists(colour_histogram_dir) or not os.path.exists(sift_descriptor_dir):
        print("Feature data not found. Extracting colour histograms and SIFT descriptors...")
        extract_preprocessed_images(preprocessed_dir)
    else:
        print(f"Using existing feature data from '{colour_histogram_dir}' and '{sift_descriptor_dir}'")

    # Step 6: Match Against Database
    print("\nMatching user image against database...")
    user_hist = extract_colour_histogram(processed_image)
    # user_edges = extract_canny_edges(processed_image)

    top_matches = match_user_features(user_hist, processed_image, top_n=5)


    display_anime_results(top_matches)

    print("\nTop matches based on colour and SIFT similarity:")
    for idx, (anime_name, combined_score) in enumerate(top_matches, 1):
        print(f"{idx}. {anime_name} - Combined Score: {combined_score:.2f}%")







        # print("\nTop matches based on colour similarity:")
        # for idx, (anime_name, dist) in enumerate(top_matches, 1):
        #     print(f"{idx}. {anime_name} - Distance: {dist:.4f}")


    # Cluster Database Images by Dominant Colour (Before Preprocessing)
    # base_dir = "C:/Users/mayom/Downloads/archive/data/anime_images"
    # print("Collecting image paths for clustering...")
    # image_paths = get_all_image_paths(base_dir)
    #
    # if image_paths:
    #     print("Running clustering on database images...")
    #     clusters = group_images_by_dominant_colour(image_paths, final_clusters=20, k_per_image=3)
    #     print("Clustering completed. Cluster summary:")
    #     # Save clusters to folder for visual inspection
    #     save_clusters(clusters, output_folder="clusters")
    #     for cluster_id, paths in clusters.items():
    #         print(f"  Cluster {cluster_id}: {len(paths)} images")
    # else:
    #     print("No images found for clustering.")




if __name__ == "__main__":
    main()