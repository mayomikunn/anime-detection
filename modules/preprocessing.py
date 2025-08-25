import cv2


def preprocess_image(user_image_path):
      # Load the image
    img = cv2.imread(user_image_path)
    if img is None:
        print("Error: Could not load image.")
        return None

      # Resize image to standard dimensions (256x256)
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

      # Apply Gaussian blur for noise reduction
    img_denoised = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # print(img_normalized)
    # print(img)
    # print(img.dtype)
    cv2.imwrite("input_preprocessed.jpg", img_denoised) # Saves preprocessed input image
    return img_denoised  # Processed image as NumPy array





