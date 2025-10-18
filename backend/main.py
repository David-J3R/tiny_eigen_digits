from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
    allow_methods=["*"],
    allow_headers=["*"],
)


# Consistent Health Check Endpoint
@app.get("/", include_in_schema=False)
def root():
    return FileResponse("frontend/index.html")


# Include Routers
app.include_router(
    predict_router,
    prefix=config.API_V1_PREFIX,
    tags=["Predictions"],
)
