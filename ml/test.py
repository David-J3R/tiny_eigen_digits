import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(__file__)  # Gets the directory where the script is
dataset_path = os.path.join(script_dir, "data")
models_path = os.path.join(script_dir, "models")
processed_path = os.path.join(script_dir, "processed_data")


def test_data_transformation():
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

    # test
    x = flattened_array
    print("min, max:", float(x.min()), float(x.max()))  # expect ~0.0, ~1.0
    print("mean:", float(x.mean()))
    print(
        ">0.5 fraction:", float((x > 0.5).mean())
    )  # should be small-ish for thin strokes
    print("nonzero fraction:", float((x > 0.0).mean()))

    img.show(title="Original")
    img_gray.show(title="Grayscale")
    img_resized.show(title="Resized 28x28")

    img28 = (x.reshape(28, 28) * 255).astype("uint8")
    Image.fromarray(img28).show(title="Reconstructed from flattened array")


# ---- Test CNN Model Prediction ----
def test_cnn_model_prediction():
    model = tf.keras.models.load_model(
        os.path.join(script_dir, "models", "cnn_best.keras")
    )
    print(model.summary())

    # Load test data
    X = np.load(os.path.join(processed_path, "X_data.npy"))
    y = np.load(os.path.join(processed_path, "y_data.npy"))

    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28, 1)
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        X = X[..., None]
    else:
        raise ValueError(f"Unexpected shape for X_test: {X.shape}")

    if X.max() > 1.0:
        X = X / 255.0

    # replicate test data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # print("Making predictions on test set")
    predictions = model.predict(X_test)
    # The output of the model is an array of probabilities for each class.
    # We use np.argmax to get the class with the highest probability.
    predicted_labels = np.argmax(predictions, axis=1)

    # Get 15 random test samples to display
    num_samples = 15
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        idx = sample_indices[i]

        # Display the image
        ax.imshow(X_test[idx], cmap="gray")

        # Set the title with the prediction and true label
        true_label = y_test[idx]
        pred_label = predicted_labels[idx]

        # Color the title green if correct, red if incorrect
        title_color = "green" if pred_label == true_label else "red"
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}", color=title_color, fontsize=14
        )

        # Remove azis ticks for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Model Predictions on Test Image", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    test_cnn_model_prediction()
