from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .routers.cnn_predict import router as predict_router

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Consistent Health Check Endpoint
@app.get("/", tags=["Health Check"])
def root():
    return {"status": "ok", "message": f"{config.APP_NAME} is running."}


# Include Routers
app.include_router(
    predict_router,
    prefix=config.API_V1_PREFIX,
    tags=["Predictions"],
)
