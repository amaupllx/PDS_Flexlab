import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# File path
file_path = "data_layer/Clamp_Stack.tif"

# Resolution definitions
resolution_area = (10**(-6)) * 0.034199899999999984**2  # m²/pixel
resolution_length = 0.034199899999999984   # mm/pixel

# Additional area approximations
R_out = 0.0503  # m
R_in = 0.0423   # m
h = 12 * 10**(-3)  # m
approx_area = 3.14 * (R_out**2 - R_in**2) * 10**4  # cm²

# Initialize a list to store the results
results = []

# Open the TIFF file using Pillow
with Image.open(file_path) as img:
    # Check the number of layers/pages
    num_layers = getattr(img, "n_frames", 1)  # `n_frames` gives the number of frames/layers
    print(f"Number of layers in the .tif file: {num_layers}")

    # Iterate through each layer in the .tif file
    for layer in range(num_layers):
        img.seek(layer)  # Go to the specific layer/frame
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

        # Count black pixels and calculate black area
        black_pixel_count = np.sum(thresh == 0)
        black_area = black_pixel_count * resolution_area * 10**4  # cm²

        # Append results to the list
        results.append({"Layer": layer + 1, "Area_cm2": black_area})

df_layers = pd.DataFrame(results)

df_layers['Height_mm'] = (df_layers['Layer']-1)*5*resolution_length # in mm

# Plot the Area (cm²) against Height (mm)
plt.figure(figsize=(10, 6))
plt.plot(df_layers['Height_mm'], df_layers['Area_cm2'], marker='o', linestyle='-', color='b')
plt.title("Area of the ring vs disctance from top")
plt.xlabel("Distance from top ring (mm)")
plt.ylabel("Area (cm²)")
plt.axvline(x=12, color='r', linestyle='--', label='Bottom of the ring\n x = 12mm')
plt.legend()
plt.grid(True)
plt.show()

# Save the results to a CSV file
df_layers.to_csv("area_layers.csv", index=False)

df_filtered = df_layers[df_layers['Height_mm'] < 12]

height_interval = 5 * resolution_length /10  # cm per layer

volume = np.sum(df_filtered['Area_cm2'] * height_interval)

print(f"Volume of the ring: {volume:.2f} cm³")

R_out = 0.0503                                      # m
R_in = 0.0423                                       # m
h = 12*10**(-3)                                     # m
approx_area = 3.14*(R_out**2 - R_in**2)*10**4             # cm2
approx_vol = approx_area * h * 100
print(f"Approx volume of the ring: {approx_vol:.2f} cm³")