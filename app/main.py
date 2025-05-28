# ad_api/app/main.py

import logging
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager # For lifespan manager

# Import configurations and services
from app.core import config
# from app.services.processing_service import ProcessingService # Will import dynamically in lifespan

# Routers will be imported here later
# from app.api.routers import system_status, streaming

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Lifespan Context Manager ---
# This will manage the startup and shutdown logic for our application resources.
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # --- Code to execute before the application starts accepting requests ---
    logger.info("FastAPI application startup sequence initiated via lifespan manager...")
    
    # Dynamically import ProcessingService here to manage dependencies
    # from app.services.processing_service import ProcessingService # Assuming this will be our service class
    
    # Initialize and start the core processing service
    logger.info("Initializing ProcessingService...")
    # For now, let's assume processing_service_instance will be stored on the app state
    # or managed in a way accessible by request handlers if needed.
    # If only startup/shutdown uses it, local variable here is fine.
    
    # app_instance.state.processing_service = ProcessingService() # Store on app state
    # logger.info("ProcessingService instance created.")
    # app_instance.state.processing_service.start_processing()
    # logger.info("ProcessingService started.")
    # For now, just a message:
    logger.info("LIFESPAN STARTUP: TODO: Initialize and start ProcessingService here.")
    
    yield # This is where the application runs (accepts requests)
    
    # --- Code to execute after the application has finished handling requests (on shutdown) ---
    logger.info("FastAPI application shutdown sequence initiated via lifespan manager...")
    # if hasattr(app_instance.state, 'processing_service') and app_instance.state.processing_service:
    #     logger.info("Stopping ProcessingService...")
    #     app_instance.state.processing_service.stop_processing()
    #     logger.info("ProcessingService stopped.")
    # For now, just a message:
    logger.info("LIFESPAN SHUTDOWN: TODO: Stop ProcessingService here.")


# Create the FastAPI application instance and assign the lifespan manager
app = FastAPI(
    title="Intelligent Ad System API",
    description="API for controlling and monitoring an intelligent ad display system.",
    version="0.1.0",
    lifespan=lifespan # Assign the lifespan context manager
)

# --- API Routers ---
# We will include these once they are created.
# Example:
# app.include_router(system_status.router, prefix="/api/v1/status", tags=["Status & Monitoring"])
# app.include_router(streaming.router, prefix="/api/v1/stream", tags=["Video Streams"])
logger.info("TODO: Include API routers here once created.")


# --- Basic Root Endpoint for Testing ---
@app.get("/", tags=["General"])
async def read_root():
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the Intelligent Ad System API! (Lifespan active)"}


# --- Main entry point for running with Uvicorn directly (for development) ---
if __name__ == "__main__":
    # Get the string name of the log level
    log_level_str = logging.getLevelName(config.LOG_LEVEL).lower()

    logger.info(f"Starting Uvicorn server directly from app/main.py on http://{config.API_HOST}:{config.API_PORT} "
                f"with log level: {log_level_str}")
    uvicorn.run(
        "app.main:app", 
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=log_level_str # Use the string representation
    )