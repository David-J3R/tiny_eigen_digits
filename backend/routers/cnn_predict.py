import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..models.cnn_predict import PredictRequest, PredictResponse
from ..models.model_loader import model
from ..utils.image_preprocessor import ImagePreprocessor

# Preprocessor dependency
preprocessor = ImagePreprocessor()


def get_preprocessor():
    # Dependency injection for the image preprocessor
    return preprocessor


# --- Set up API Router ---
router = APIRouter(tags=["Predict CNN"])


# --- API Endpoints ---


@router.get("/")
def read_root():
    """A simple root endpoint to verify the API is running."""
    return {"status": "ok", "message": "Digit Recognizer API is running."}


@router.post("/predict", response_model=PredictResponse)
def predict_digit(
    drawing: PredictRequest,
    img_preprocessor: ImagePreprocessor = Depends(get_preprocessor),
):
    """
    Prediction endpoint.
    Receives a base64 encoded image, preprocesses it,
    and returns the model's prediction.
    """
    # STEP 1: Preprocess
    try:
        processed_img = img_preprocessor.process(drawing.image_base64)
    except Exception as e:
        # Handle preprocessing errors
        raise HTTPException(status_code=400, detail=str(e))

    # STEP 2: Model Prediction
    try:
        prediction = model.predict(processed_img)
        predicted_digit = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction, axis=1)[0])

        # Pydantic model for automatic validation
        return PredictResponse(predicted_digit=predicted_digit, confidence=confidence)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction: {e}",
        )
