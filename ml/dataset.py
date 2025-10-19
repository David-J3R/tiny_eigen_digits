import os

import numpy as np
from PIL import Image
from tqdm import tqdm

"""
    Loads images from the specified path and processes them into a dataset.

    This function implements the 5 key preprocessing steps:
    1. Load Image
    2. Convert to Grayscale
    3. Resize to 28x28
    4. Normalize pixel values to 0-1
    5. Flatten the image into a 1D vector (Useless for CNN, but included for completeness)
"""


def create_dataset(dataset_path):
    print(f"Reading images from: {dataset_path}")

    # Initialize lists to hold the data and labels
    X_data = []
    Y_data = []

    # The kaggle Handwritten zip dataset is divide into 9 different folders
    # for each number
    # Loop through each folder named from 0 to 9
    for digit in map(str, range(10)):
        digit_path = os.path.join(dataset_path, digit)
        print(f"Folder {digit_path} founded")

        # Skip any non-directory files
        if not os.path.isdir(digit_path):
            continue

        # Loop through each image in the digit's folder, with a progress bar (using tqdm)
        # os.listdir() returns filenames in arbitrary filesystem order.
        for image_file in tqdm(
            os.listdir(digit_path), desc=f"Processing digit {digit}"
        ):
            image_path = os.path.join(digit_path, image_file)

            try:
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
                # float32 for memory-efficient

                # STEP 5: Flatten the 28x28 array into a 1D vector of 784 features
                # SUPER USELESS STEP since we will use CNN, but I didn't remove it XD
                # I want to see if someone is paying attention
                flattened_array = normalized_array.flatten()
                # print(flattened_array)
                # break

                # Add the processed data and its label to our lists
                X_data.append(flattened_array)
                Y_data.append(int(digit))

            except Exception as e:
                print(f"Error: Could not process image {image_path}. Error: {e}")

    return np.array(X_data), np.array(Y_data)


# --- Main execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)  # Gets the directory where the script is
    dataset_path = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, "processed_data")

    # Create the dataset
    X, y = create_dataset(dataset_path)
    print("\n--- Data Processing Summary ---")
    print(f"Total images processed: {len(X)}")
    print(f"Shape of the feature matrix (X): {X.shape}")
    print(f"Shape of the labels vector (y): {y.shape}")

    # Save the processed dataset to .npy files for later use
    np.save(os.path.join(output_dir, "X_data.npy"), X)
    np.save(os.path.join(output_dir, "y_data.npy"), y)
    print(
        "\nProcessed data and labels have been saved to the 'processed_data' directory."
    )

    # Display one of the processed images to verify
    import matplotlib.pyplot as plt

    print("\nDisplaying a random sample from the processed data...")
    sample_index = np.random.randint(0, len(X))
    plt.imshow(X[sample_index].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {y[sample_index]}")
    plt.show()
