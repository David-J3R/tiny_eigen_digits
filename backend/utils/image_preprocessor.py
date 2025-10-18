import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """ "
    Encapsulates the logic to preprocess to match the MNIST dataset format.
    Format which was use for training the CNN model.
    Goal: it isn't improve the preprocessing training, but to perfectly replicate it.
    """

    def __init__(self, img_size: Tuple[int, int] = (28, 28)):
        # Initializes the preprocessor configuration.
        self.img_size = img_size

    def process(self, base64_str: str) -> np.ndarray:
        # --- Main preprocessing pipeline ---
        """
        The steps are strictly aligned with the training preprocessing.
        """

        # STEP 1 & 2 (Combined): Decode and convert to grayscale
        img = self._decode_base64(base64_str)

        # STEP 3: Resize to 28x28 (brute-force, like in training)
        img_resized = img.resize(self.img_size, Image.Resampling.LANCZOS)

        # Convert to NumPy array
        pixel_array = np.array(img_resized, dtype=np.float32)

        # Invert the colors (like in training)
        # copying MNIST format
        inverted_array = 255.0 - pixel_array

        # STEP 4: Normalize pixel values to [0, 1]
        normalized_array = inverted_array / 255.0

        # STEP 5: Reshape to (1, 28, 28, 1) for model input
        return normalized_array.reshape(1, *self.img_size, 1)

    def _decode_base64(self, base64_str: str) -> Image.Image:
        # Decodes a base64 string to a PIL Image in grayscale mode.
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        try:
            # Decode the base64 string
            img_bytes = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(img_bytes)).convert("L")
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}") from e
