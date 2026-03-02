"""
Flask API for Deepfake Detection
REST API endpoints for video upload and real-time detection
"""

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    print("[WARNING] flask-cors not installed. CORS support disabled.")
    print("[INFO] Install with: pip install flask-cors")
    CORS_AVAILABLE = False

import os
import tempfile
import torch
from pathlib import Path
import uuid
from inference import DeepfakeDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)  # Enable CORS for frontend requests
else:
    logger.warning("CORS support disabled - install flask-cors for cross-origin requests")

# Configuration
CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT', 'checkpoints/best_model.pth')
UPLOAD_FOLDER = tempfile.gettempdir()
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

# Auto-detect device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize detector lazily
detector = None


def get_detector():
    """Lazily initialize the detector on first use."""
    global detector
    if detector is None:
        logger.info(f"Initializing detector with checkpoint: {CHECKPOINT_PATH} on {DEVICE}")
        try:
            detector = DeepfakeDetector(
                checkpoint_path=CHECKPOINT_PATH,
                device=DEVICE,
                debug=False
            )
            logger.info("Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise
    return detector


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'device': DEVICE,
        'checkpoint': CHECKPOINT_PATH
    })


@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    """
    Main detection endpoint - Upload video for analysis

    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: video file with key 'video'

    Response:
        {
            "success": true,
            "prediction": "FAKE" or "REAL",
            "confidence": 88.5,
            "fake_probability": 0.885,
            "real_probability": 0.115,
            "processing_time": 2.34,
            "video_id": "unique-id",
            "message": "Analysis complete"
        }
    """
    # Check if file is present
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file provided'
        }), 400

    file = request.files['video']

    # Check if file is valid
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({
            'success': False,
            'error': f'File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB'
        }), 400

    temp_path = None
    try:
        det = get_detector()

        # Save uploaded file
        video_id = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.{ext}")

        logger.info(f"Saving uploaded video: {temp_path}")
        file.save(temp_path)

        # Run detection
        logger.info(f"Running detection on video_id: {video_id}")
        results = det.detect_from_video_file(temp_path)

        # Cleanup
        os.remove(temp_path)
        logger.info(f"Removed temporary file: {temp_path}")

        # Format response
        response = {
            'success': True,
            'video_id': video_id,
            'prediction': results['prediction'],
            'confidence': round(results['confidence'], 2),
            'fake_probability': round(results['fake_probability'], 4),
            'real_probability': round(results['real_probability'], 4),
            'processing_time': round(results['processing_time'], 3),
            'timestamp': results['timestamp'],
            'message': 'Analysis complete'
        }

        logger.info(f"Detection complete: {response['prediction']} ({response['confidence']}%)")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during detection: {str(e)}", exc_info=True)

        # Cleanup on error
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """
    Batch detection endpoint - Upload multiple videos

    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: multiple video files with key 'videos'

    Response:
        {
            "success": true,
            "results": [...],
            "total_videos": 5,
            "total_processing_time": 12.5
        }
    """
    if 'videos' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No videos provided'
        }), 400

    files = request.files.getlist('videos')

    if len(files) == 0:
        return jsonify({
            'success': False,
            'error': 'No videos provided'
        }), 400

    det = get_detector()
    results_list = []
    total_time = 0

    for file in files:
        if not allowed_file(file.filename):
            results_list.append({
                'filename': file.filename,
                'success': False,
                'error': 'Invalid file type'
            })
            continue

        temp_path = None
        try:
            # Save and process
            video_id = str(uuid.uuid4())
            ext = file.filename.rsplit('.', 1)[1].lower()
            temp_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.{ext}")

            file.save(temp_path)
            results = det.detect_from_video_file(temp_path)
            os.remove(temp_path)

            results_list.append({
                'video_id': video_id,
                'filename': file.filename,
                'success': True,
                'prediction': results['prediction'],
                'confidence': round(results['confidence'], 2),
                'fake_probability': round(results['fake_probability'], 4),
                'real_probability': round(results['real_probability'], 4),
                'processing_time': round(results['processing_time'], 3)
            })

            total_time += results['processing_time']

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)
            results_list.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'results': results_list,
        'total_videos': len(files),
        'total_processing_time': round(total_time, 3)
    }), 200


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'checkpoint_path': CHECKPOINT_PATH,
        'device': DEVICE,
        'model_loaded': detector is not None,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024)
    })


if __name__ == '__main__':
    # Development server - use debug=False for production
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
