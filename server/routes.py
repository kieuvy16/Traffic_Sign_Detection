"""
API Routes for Traffic Sign Detection System
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
from typing import List
import yaml

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Import models (will be injected via dependency injection)
yolo_detector = None
cnn_classifier = None
config = None

def setup_routes(app_yolo_detector, app_cnn_classifier, app_config):
    """Setup routes with injected dependencies"""
    global yolo_detector, cnn_classifier, config
    yolo_detector = app_yolo_detector
    cnn_classifier = app_cnn_classifier
    config = app_config

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "yolo_model_loaded": yolo_detector is not None,
        "cnn_model_loaded": cnn_classifier is not None,
        "version": "1.0.0"
    }

@router.post("/detect")
async def detect_traffic_signs(file: UploadFile = File(...)):
    """
    Detect traffic signs in an uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
    
    Returns:
        JSON with detection results
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check file size
        if len(contents) > config['processing']['max_image_size']:
            raise HTTPException(status_code=400, detail="Image size too large")
        
        # Detect with YOLO
        if yolo_detector is None:
            raise HTTPException(status_code=500, detail="YOLO model not loaded")
        
        conf_threshold = config['models']['yolo']['confidence']
        iou_threshold = config['models']['yolo']['iou_threshold']
        
        results = yolo_detector.detect(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Classify each detection with CNN (optional refinement)
        enhanced_detections = []
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            
            # Crop detected region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0 and cnn_classifier is not None:
                try:
                    # Classify with CNN
                    cnn_result = cnn_classifier.predict(cropped)
                    
                    det['cnn_prediction'] = {
                        'class_id': cnn_result['class_id'],
                        'confidence': cnn_result['confidence']
                    }
                except Exception as e:
                    logger.warning(f"CNN classification failed for detection: {e}")
            
            enhanced_detections.append(det)
        
        return JSONResponse(content={
            "success": True,
            "num_detections": len(enhanced_detections),
            "detections": enhanced_detections,
            "image_size": {
                "height": image.shape[0],
                "width": image.shape[1]
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect_batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Detect traffic signs in multiple images
    
    Args:
        files: List of image files
    
    Returns:
        JSON with detection results for each image
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image file"
                })
                continue
            
            # Detect
            detections = yolo_detector.detect(image)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detections
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_images": len(files),
        "results": results
    })

@router.get("/classes")
async def get_classes():
    """Get list of supported traffic sign classes"""
    try:
        return JSONResponse(content={
            "success": True,
            "num_classes": len(config['classes']['names']),
            "classes": config['classes']['names']
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify")
async def classify_sign(file: UploadFile = File(...)):
    """
    Classify a cropped traffic sign image using CNN
    
    Args:
        file: Cropped traffic sign image
    
    Returns:
        Classification result
    """
    try:
        if cnn_classifier is None:
            raise HTTPException(status_code=500, detail="CNN model not loaded")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Classify
        result = cnn_classifier.predict(image)
        
        return JSONResponse(content={
            "success": True,
            "prediction": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        return JSONResponse(content={
            "success": True,
            "config": config
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "yolo_model_loaded": yolo_detector is not None,
            "cnn_model_loaded": cnn_classifier is not None,
            "num_classes": len(config['classes']['names']) if config else 0,
            "max_image_size": config['processing']['max_image_size'] if config else 0
        }
        return JSONResponse(content={
            "success": True,
            "stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))