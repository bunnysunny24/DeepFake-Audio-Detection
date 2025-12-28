"""
🧪 Test Inference Script
Quick testing of inference functionality
"""

import sys
from pathlib import Path
from inference import DeepfakeDetector


def test_video_file(checkpoint_path: str, video_path: str):
    """Test detection on a video file"""
    
    print("\n" + "="*80)
    print("🧪 TESTING VIDEO FILE INFERENCE")
    print("="*80)
    
    # Initialize detector
    print("\n1️⃣ Initializing detector...")
    detector = DeepfakeDetector(
        checkpoint_path=checkpoint_path,
        device='cuda',
        quantized=False,
        debug=True  # Show component contributions
    )
    
    # Run detection
    print("\n2️⃣ Running detection...")
    results = detector.detect_from_video_file(video_path)
    
    print("\n3️⃣ Test complete!")
    print(f"✅ Video: {video_path}")
    print(f"✅ Prediction: {results['prediction']}")
    print(f"✅ Confidence: {results['confidence']:.2f}%")
    print(f"✅ Processing time: {results['processing_time']:.3f}s")
    
    return results


def test_api():
    """Test Flask API"""
    
    print("\n" + "="*80)
    print("🌐 TESTING FLASK API")
    print("="*80)
    
    import requests
    import json
    
    api_url = "http://localhost:5000"
    
    # 1. Health check
    print("\n1️⃣ Testing health endpoint...")
    response = requests.get(f"{api_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # 2. Model info
    print("\n2️⃣ Testing model info endpoint...")
    response = requests.get(f"{api_url}/api/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # 3. Detection (requires video file)
    video_path = input("\nEnter path to test video (or press Enter to skip): ").strip()
    
    if video_path and Path(video_path).exists():
        print("\n3️⃣ Testing detection endpoint...")
        
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f"{api_url}/api/detect", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        print("\n✅ API test complete!")
    else:
        print("\n⚠️ Skipping detection test (no video provided)")


def main():
    """Main test function"""
    
    print("\n" + "="*100)
    print("🎯 DEEPFAKE DETECTION INFERENCE TEST SUITE")
    print("="*100)
    
    # Get checkpoint path
    checkpoint_path = input("\nEnter path to model checkpoint (.pth file): ").strip()
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Test type
    print("\nSelect test type:")
    print("1. Test video file inference")
    print("2. Test Flask API (requires server running)")
    print("3. Test both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        # Test video file
        video_path = input("\nEnter path to test video: ").strip()
        
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
        else:
            test_video_file(checkpoint_path, video_path)
    
    if choice in ['2', '3']:
        # Test API
        print("\n⚠️ Make sure Flask server is running (python inference_api.py)")
        input("Press Enter when ready...")
        test_api()
    
    print("\n" + "="*100)
    print("✅ ALL TESTS COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
