import os
import magic
import sys

def get_user_image():
    user_image_path = input("Enter the image file path: ")

    if not os.path.isfile(user_image_path):
        print("Error: The file does not exist.")
        sys.exit(1)

    if is_valid_image(user_image_path):
        return user_image_path

    else:
        print("Error: Invalid file format. Please upload a PNG, JPG, or JPEG file.")
        sys.exit(1)




def is_valid_image(user_image_path):
    # Checks if the given file is a valid image by checking its MIME type.
    # image_path: Path to the uploaded file.
    # return true if the file is an image, false otherwise.

    try:
        mime_type = magic.from_file(user_image_path, mime=True)
        return mime_type.startswith("image/")
    except Exception as e:
        print(f"Error checking file type: {e}")
        return False