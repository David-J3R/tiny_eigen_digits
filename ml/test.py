import os

import numpy as np
from PIL import Image

script_dir = os.path.dirname(__file__)  # Gets the directory where the script is
dataset_path = os.path.join(script_dir, "data")
output_dir = os.path.join(script_dir, "processed_data")


digit = "0"
digit_path = os.path.join(dataset_path, digit)

for image_file in os.listdir(digit_path):
    image_path = os.path.join(digit_path, image_file)
    # print(image_file)

# print(image_path)

# STEP 1: Load the Image
img = Image.open(image_path)
# STEP 2: Convert to Grayscale
img_gray = img.convert("L")

# STEP 3: Resize to 28x28
img_resized = img_gray.resize((28, 28))

# Convert the image to a NumPy array
pixel_array = np.array(img_resized)
# print(pixel_array)
# break

# Image with Light background and Dark ink
# Invert the colors so that the digit is represented by higher values (closer to 1)
inverted_array = 255.0 - pixel_array
# display(Image.fromarray(inverted_array))
# break

# STEP 4: Normalize pixel values to the 0.0 to 1.0 range
normalized_array = (inverted_array / 255.0).astype(np.float32)

# STEP 5: Flatten the 28x28 array into a 1D vector of 784 features
flattened_array = normalized_array.flatten()


##############test
x = flattened_array
print("min, max:", float(x.min()), float(x.max()))  # expect ~0.0, ~1.0
print("mean:", float(x.mean()))
print(">0.5 fraction:", float((x > 0.5).mean()))  # should be small-ish for thin strokes
print("nonzero fraction:", float((x > 0.0).mean()))


img.show(title="Original")
img_gray.show(title="Grayscale")
img_resized.show(title="Resized 28x28")


img28 = (x.reshape(28, 28) * 255).astype("uint8")
Image.fromarray(img28).show(title="Reconstructed from flattened array")
