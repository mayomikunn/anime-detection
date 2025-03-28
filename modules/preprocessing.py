import cv2

def preprocess_image(user_image_path):
      # Load the image
    img = cv2.imread(user_image_path)
    if img is None:
        print("Error: Could not load image.")
        return None

      # Resize image to standard dimensions (256x256)
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

       # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur for noise reduction
    img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # print(img_normalized)
    # print(img)
    # print(img.dtype)
    return img_denoised  # Processed image as NumPy array
