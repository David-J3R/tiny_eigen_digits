# Upgrading from os to pathlib for better path handling
# PathLib is more robust and cross-platform
from pathlib import Path

import tensorflow as tf

# Path to trained model file
MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "models" / "cnn_best.keras"

# Load once at import time
model = tf.keras.models.load_model(MODEL_PATH.as_posix())
