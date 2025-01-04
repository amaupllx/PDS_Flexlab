from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# File path
file_path = "data_layer/Clamp_Stack.tif"


resolution_area = (10**(-6))*0.034199899999999984**2     # m²/pixel
resolution_length = (10**(-3))*0.034199899999999984      # m/pixel

# Additional area approximations
R_out = 0.0503  # m
R_in = 0.0423   # m
h = 12 * 10**(-3)  # m
approx_area = 3.14 * (R_out**2 - R_in**2) * 10**4  # cm²

# Open the TIFF file using Pillow
with Image.open(file_path) as img:
    # Check the number of layers/pages
    num_layers = getattr(img, "n_frames", 1)  # `n_frames` gives the number of frames/layers
    print(f"Number of layers in the .tif file: {num_layers}")

    # Read and process the first layer
    img.seek(35)  # Go to the first frame
    first_layer = np.array(img)

    # Check if the image is grayscale or color
    if len(first_layer.shape) == 3:  # If 3D array, it's a color image
        img_gray = cv2.cvtColor(first_layer, cv2.COLOR_BGR2GRAY)
    else:  # If 2D array, it's already grayscale
        img_gray = first_layer

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # Apply binary inverse thresholding
    ret, thresh = cv2.threshold(img_blur, 160, 255, cv2.THRESH_BINARY_INV)

    # Count white and black pixels
    black_pixel_count = np.sum(thresh == 0)

    # Calculate areas
    black_area = black_pixel_count * resolution_area * 10**4  # cm²

    # Print results
    print(f"Black area: {black_area:.2f} cm²")
    # Generate real-world axis values
    height, width = thresh.shape
    x_real = np.arange(0, width) * resolution_length * 100  # cm
    y_real = np.arange(0, height) * resolution_length * 100  # cm

    # Plot the image with real-world axis
    plt.imshow(thresh, cmap="gray", extent=[x_real[0], x_real[-1], y_real[-1], y_real[0]])
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.show()

    '''
    # Use a resizable window
    cv2.namedWindow('Binary Inverse Threshold', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Inverse Threshold', thresh)

    # Resize the window to fit the screen (optional)
    screen_res = (1280, 720)  # Example screen resolution (width x height)
    scale_width = screen_res[0] / thresh.shape[1]
    scale_height = screen_res[1] / thresh.shape[0]
    scale = min(scale_width, scale_height)  # Scale to fit the screen
    window_width = int(thresh.shape[1] * scale)
    window_height = int(thresh.shape[0] * scale)
    cv2.resizeWindow('Binary Inverse Threshold', window_width, window_height)

    cv2.waitKey(0)  # Wait for a key press to close the image
    cv2.destroyAllWindows()
'''
print(f"Approx area: {approx_area:.2f} cm²")