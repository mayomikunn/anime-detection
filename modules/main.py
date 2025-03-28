from user_input import get_user_image
from database import load_database_images
from preprocessing import preprocess_image
import os

def main():
    # Get the image path from user
    user_image_path = get_user_image()

    # Preprocess the user image
    processed_image = preprocess_image(user_image_path)

    # Check if preprocessed data file exists before re-processing everything
    preprocessed_file = "processed_images"
    if not os.path.exists(preprocessed_file):
        print("Preprocessed data not found. Running preprocessing...")
        processed_images = load_database_images()  # Adjust parameters as needed
    else:
        print(f"Using existing preprocessed data from {preprocessed_file}")



    if processed_image is not None:
        print("User image preprocessed successfully!")

if __name__ == "__main__":
    main()
