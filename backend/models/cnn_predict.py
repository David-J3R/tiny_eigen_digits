from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str  # e.g. "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."


# Pydantic model for the response for auto-documentation
class PredictResponse(BaseModel):
    predicted_digit: int = Field(
        ...,
        description="The digit predicted by the model (0-9).",
        examples=[7],
    )
    confidence: float = Field(
        ...,
        description="The model's confidence score for the prediction (0.0-1.0).",
        examples=[0.998],
    )
