import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


class CNN_DigitClassifier:
    """
    A class to build, train, evaluate, and save a CNN for digit classification.
    It incorporates Batch Normalization, Callbacks, and the Keras Functional API.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.history = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)
        self.num_classes = None

        # ---- Path Definitions ----
        script_dir = os.path.dirname(__file__)
        self.processed_data_path = os.path.join(script_dir, "processed_data")
        self.model_output_dir = os.path.join(script_dir, "models")
        os.makedirs(
            self.model_output_dir, exist_ok=True
        )  # Ensure the model output directory exists

        # ---- Set random seeds for reproducibility ----
        tf.keras.utils.set_random_seed(self.seed)
        np.random.seed(self.seed)

    def load_data(self, test_size=0.2):
        """
        Loads, reshapes, normalizes, splits, and one-hot encodes the dataset.
        """
        print("--- Starting Data Preparation ---")
        # 1. Load the processed dataset
        # shape: (N, 784)
        X = np.load(os.path.join(self.processed_data_path, "X_data.npy"))
        # shape: (N,)
        y = np.load(os.path.join(self.processed_data_path, "y_data.npy"))

        # 2. Reshape and normalize X for CNN input
        # A CNN expects a 4D tensor: (samples, height, width, channels)
        # Our images are 28x28 and have 1 channel (grayscale)
        # Ensure shape, dtype, scaling
        if X.ndim == 2 and X.shape[1] == 784:
            # dtype and scaling already handled in dataset.py
            # the -1 is for numpy to infer the number of samples
            X = X.reshape(-1, 28, 28, 1)
        elif X.ndim == 3 and X.shape[1:] == (28, 28):
            X = X[..., None]
        else:
            raise ValueError(f"Unexpected shape for X: {X.shape}")

        # Only if preprocessing data is not yet normalized
        if X.max() > 1.0:
            X /= 255.0

        # 3. Split the dataset into training and testing sets
        # One-hot encoding later after the split to avoid data leakage
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )

        # 4. One-hot encode the labels
        self.num_classes = len(np.unique(y))
        self.y_train = keras.utils.to_categorical(
            y_train_raw, num_classes=self.num_classes
        )
        self.y_test = keras.utils.to_categorical(
            y_test_raw, num_classes=self.num_classes
        )
        self.X_train = X_train
        self.X_test = X_test

        print(
            f"Train: {self.X_train.shape}, Test: {self.X_test.shape}, Classes: {self.num_classes}"
        )
        print("--- Data Preparation Completed ---\n")

    def build_model(self):
        """
        Builds the CNN model (padding + BN + a small Dense head).
        """
        print("--- Building the CNN Model ---")
        inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Conv2D(32, 3, padding="same", activation=None)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.Conv2D(64, 3, padding="same", activation=None)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.Conv2D(128, 3, padding="same", activation=None)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        # Build the CNN Model
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.summary()  # Prints a summary of the model's layers

    # --- Training with Callbacks ---
    def train(self, epochs=20, batch_size=128):
        """
        Trains the model with callbacks for early stopping and saving the best version.
        """
        if self.X_train is None:
            self.load_data()
        if self.model is None:
            self.build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_accuracy"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=2, min_lr=1e-5, monitor="val_loss"
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_output_dir, "cnn_best.keras"),
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ]

        print("\n --- Starting Model Training ---")
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,  # Verbose for detailed logging
        )  # Use 10% of training data for validation

        print("--- Model Training Completed ---\n")

    def evaluate(self):
        """
        Evaluates the final model on the test set.
        """
        print("\n --- Evaluating the Model on Test Data ---")
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("\n--- CNN Model Performance ---")
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}\n")
        return loss, accuracy

    def save(self, filename="cnn_digit_recognizer.keras"):
        """Saves the final trained model."""
        model_save_path = os.path.join(self.model_output_dir, filename)
        self.model.save(model_save_path)
        print(f"Model saved to: {model_save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Create a classifier instance
    classifier = CNN_DigitClassifier(seed=42)

    # 2. Train the model (data loading/prep and model building happen automatically)
    classifier.train()

    # 3. Evaluate the model
    classifier.evaluate()

    # 4. Save the final model
    classifier.save()
