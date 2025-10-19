# Handwritten Digit Recognizer (End-to-End Project)

A production-ready CNN model that recognizes handwritten digits from 0-9, deployed as a REST API with an interactive web interface.

**Final Accuracy: 98.63%**

## The Problem

I found a European (Swiss) Handwritten Digits dataset on Kaggle containing over 21,000 images of numbers from 0-9. I wanted to build a real model with a real use—not another Jupyter Notebook collecting dust. So over 2 weekends I trained, built, and deployed a complete end-to-end digit recognizer.

The challenge wasn't just building a model that could hit 90% accuracy. It was engineering a robust preprocessing pipeline to turn those 21,000+ images into an MNIST-lookalike dataset, then deploying it properly with FastAPI, Docker, and a clean web interface.

## What I Built

- **Trained and evaluated a CNN in TensorFlow** to achieve 98.63% accuracy on the dataset
- **Built a production-ready, versioned REST API** using FastAPI to serve the model's predictions
- **Engineered a robust image preprocessing pipeline** in Python (NumPy, PIL) to ensure inference data perfectly matched the model's training data
- **Developed a lightweight, interactive web interface** (HTML, CSS, JavaScript) to provide a real-time, user-facing demo of the model
- **Implemented professional backend practices** including dependency injection, configuration management, and robust error handling
- **Dockerized the entire stack** for easy deployment and testing

## Project Structure

```
Tiny MLOps Eigen Digits/
├── backend/                    # FastAPI application
│   ├── main.py                # App initialization, CORS, routes
│   ├── config.py              # Environment-based configuration
│   ├── models/
│   │   ├── cnn_predict.py     # Pydantic request/response models
│   │   └── model_loader.py    # Model loaded once at startup
│   ├── routers/
│   │   └── cnn_predict.py     # API endpoints
│   └── utils/
│       └── image_preprocessor.py  # Preprocessing pipeline
├── ml/                         # Machine Learning pipeline
│   ├── dataset.py             # Data preprocessing script
│   ├── model.py               # CNN architecture and training
│   ├── test.py                # Model testing and visualization
│   ├── data/                  # Training dataset (21,000+ images)
│   ├── models/                # Trained models
│   └── processed_data/        # Preprocessed numpy arrays
├── frontend/
│   └── index.html             # Single-page drawing application (Frontend is just the interactive wrapper)
├── docker/
│   └── Dockerfile             # Container configuration
└── requirements.txt           # Production dependencies
```

## The Model

**CNN Architecture:**
- Conv2D (32 filters) → BatchNorm → ReLU → MaxPool
- Conv2D (64 filters) → BatchNorm → ReLU → MaxPool
- Conv2D (128 filters) → BatchNorm → ReLU
- Flatten → Dropout(0.5) → Dense(128) → Dense(10, Softmax)

**Training:**
- Adam optimizer (lr=0.001)
- Categorical crossentropy loss
- EarlyStopping (patience=5, monitor val_accuracy)
- ReduceLROnPlateau (factor=0.5, patience=2)
- ModelCheckpoint (save_best_only=True)

**Data:**
- 80/20 train-test split (stratified)
- 10% validation split during training
- Batch size: 128
- Epochs: 20 (with early stopping)

## The Preprocessing Pipeline

This is where the magic happens. The high accuracy (98.63%) was because of all the work in the preprocessing stage—one of the most important processes to train a good model. I followed 5 steps to turn those 21,000+ images into an MNIST-lookalike dataset:

1. **Load Image** - PIL.Image.open()
2. **Convert to Grayscale** - Single channel (L mode)
3. **Resize to 28x28** - LANCZOS interpolation
4. **Invert Colors** - MNIST format (white background → black background)
5. **Normalize** - Pixel values to [0.0, 1.0] range

**Critical Design Decision:** The backend preprocessing pipeline ([image_preprocessor.py](backend/utils/image_preprocessor.py)) perfectly replicates the training preprocessing. Not to improve it, but to perfectly replicate it. This ensures inference data matches training data exactly—no train-test mismatch.

## API Endpoints

**Base URL:** `http://127.0.0.1:8000`

### GET `/`
Serves the frontend HTML interface

### GET `/api/v1/`
Health check endpoint
```json
{
  "status": "ok",
  "message": "Digit Recognizer API is running."
}
```

### POST `/api/v1/predict`
Main prediction endpoint

**Request:**
```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response:**
```json
{
  "predicted_digit": 7,
  "confidence": 0.9863
}
```

## Getting Started

### Local Development

**1. Clone the repository**
```bash
git clone https://github.com/David-J3R/tiny_eigen_digits.git
cd tiny_eigen_digits
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set environment**
```bash
echo "ENV_STATE=dev" > .env
```

**4. Run the API**
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**5. Open your browser**
Navigate to `http://127.0.0.1:8000` and start drawing digits

### Docker Deployment

I built a Dockerfile in case you want to test the app for yourself.

**Build the image:**
```bash
docker build -t digit-recognizer -f docker/Dockerfile .
```

**Run the container:**
```bash
docker run -p 8000:8000 digit-recognizer
```

Access the app at `http://localhost:8000`


## Tech Stack

**Backend:**
- FastAPI - Modern web framework
- Uvicorn - ASGI server
- TensorFlow (CPU) - Deep learning inference
- Pydantic - Data validation and settings

**ML Pipeline:**
- TensorFlow/Keras - Model training
- NumPy - Numerical computing
- Pillow - Image processing
- Scikit-learn - Data splitting
- Matplotlib - Visualization

**Frontend:**
- Vanilla HTML/CSS/JavaScript - No framework dependencies
- HTML5 Canvas - Drawing interface

All-in-one HTML Frontend created using Gemini 2.5 pro.
The frontend is just the interactive "wrapper" to show it off, so adding a complex frontend doesn't add any value to my project. 

**DevOps:**
- Docker - Containerization
- Python 3.13-slim - Base image

## Configuration

The app uses environment-based configuration ([config.py](backend/config.py)) with three modes:

- **dev** - Development mode (localhost CORS)
- **prod** - Production mode (configured origins only)
- **test** - Testing mode

Set `ENV_STATE` in `.env` to switch between modes.

## Project Stats

- **Total Python Code:** 573 lines
- **Training Dataset:** ~11,000 images (10 classes)
- **Model Size:** 11MB
- **Test Accuracy:** 98.63%
- **Inference Time:** <100ms per image
- **Docker Image:** ~2.17GB (with tensorflow-cpu)

## Results & Reflections

GOAL:

## Development Tools

**Code Quality:**
```bash
pip install -r requirements-dev.txt
```
- ruff - Fast Python linter
- black - Code formatter
- isort - Import sorter
- pytest - Testing framework
- httpx - HTTP client for testing

## License

This project is open source and available for educational purposes.

## Author

David J3R

GitHub: [@David-J3R](https://github.com/David-J3R)


