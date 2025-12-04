"""
SIMPLE VIDEO TESTER - Just give it a video, it tells you FAKE or REAL
Usage: python test_single_video.py "path/to/video.mp4"
"""
import sys
import os
from pathlib import Path

def test_video(video_path):
    """Test a single video and print result."""
    
    # Check video exists
    if not os.path.exists(video_path):
        print(f"❌ ERROR: Video not found: {video_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"🎬 TESTING VIDEO")
    print(f"{'='*70}")
    print(f"📹 File: {os.path.basename(video_path)}")
    print(f"📂 Path: {video_path}")
    
    # Find the latest checkpoint
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    output_dir = Path(__file__).parent / "outputs"
    
    checkpoint_path = None
    
    # Look for best_model.pth in checkpoint directories
    if checkpoint_dir.exists():
        for run_dir in sorted(checkpoint_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                best_model = run_dir / "best_model.pth"
                if best_model.exists():
                    checkpoint_path = best_model
                    break
    
    # Also check outputs directory
    if not checkpoint_path and output_dir.exists():
        for run_dir in sorted(output_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                best_model = run_dir / "best_model.pth"
                if best_model.exists():
                    checkpoint_path = best_model
                    break
    
    if not checkpoint_path:
        print("\n❌ ERROR: No trained model found!")
        print("   Please train the model first by running:")
        print("   .\\train_combined_dataset.ps1")
        return
    
    print(f"🔧 Model: {checkpoint_path.parent.name}/{checkpoint_path.name}")
    
    # Import and run prediction
    try:
        from predict_deployment import predict_any_video
        
        print("\n🔄 Processing video...")
        result = predict_any_video(str(video_path), str(checkpoint_path), debug=False)
        
        if result is None:
            print("\n❌ Prediction failed!")
            return
        
        # Display result
        prediction = result.get('prediction', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        real_prob = result.get('real_probability', 0)
        fake_prob = result.get('fake_probability', 0)
        
        print(f"\n{'='*70}")
        print(f"🎯 RESULT")
        print(f"{'='*70}")
        
        if prediction == 'REAL':
            print(f"✅ REAL VIDEO ({confidence:.1f}% confidence)")
        elif prediction == 'FAKE':
            print(f"🚨 FAKE VIDEO ({confidence:.1f}% confidence)")
        else:
            print(f"⚠️  UNKNOWN ({confidence:.1f}% confidence)")
        
        print(f"\n📊 Probabilities:")
        print(f"   Real: {real_prob:.1f}%")
        print(f"   Fake: {fake_prob:.1f}%")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ ERROR: Please provide a video path")
        print("\nUsage:")
        print('   python test_single_video.py "path/to/video.mp4"')
        print("\nExample:")
        print('   python test_single_video.py "D:\\Videos\\test.mp4"')
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_video(video_path)
