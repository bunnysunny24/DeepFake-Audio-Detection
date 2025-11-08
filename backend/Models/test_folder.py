"""
Batch test multiple videos in a folder.
Shows progress and saves results to CSV file.
"""
import os
import sys
import glob
import csv
from datetime import datetime
from tqdm import tqdm
from predict_deployment import predict_any_video

def test_folder(folder_path, checkpoint_path="server_checkpoints/best_model.pth"):
    """Test all videos in a folder."""
    
    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not video_files:
        print(f"❌ No videos found in {folder_path}")
        return
    
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"batch_test_{timestamp}.csv")
    
    print(f"\n{'='*70}")
    print(f"📁 BATCH TESTING: {len(video_files)} videos")
    print(f"{'='*70}")
    print(f"Folder: {folder_path}")
    print(f"Results will be saved to: {output_file}")
    print(f"{'='*70}\n")
    
    results = []
    
    # Open CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'prediction', 'confidence', 'real_prob', 'fake_prob', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Use tqdm for progress bar
        with tqdm(total=len(video_files), desc="Processing videos", unit="video") as pbar:
            for i, video_path in enumerate(video_files, 1):
                video_name = os.path.basename(video_path)
                pbar.set_description(f"Processing: {video_name[:40]}")
                
                print(f"\n{'='*70}")
                print(f"📹 [{i}/{len(video_files)}] {video_name}")
                print(f"{'='*70}")
                
                result = predict_any_video(video_path, checkpoint_path)
                
                if result:
                    row = {
                        'file': video_name,
                        'prediction': result['prediction'],
                        'confidence': f"{result['confidence']:.2f}",
                        'real_prob': f"{result['real_prob']:.2f}",
                        'fake_prob': f"{result['fake_prob']:.2f}",
                        'status': 'success'
                    }
                    results.append(row)
                    writer.writerow(row)
                    csvfile.flush()  # Write immediately to file
                    
                    # Show quick result
                    emoji = "🚨" if result['prediction'] == 'FAKE' else "✅"
                    tqdm.write(f"{emoji} Result: {result['prediction']} ({result['confidence']:.1f}%)")
                else:
                    row = {
                        'file': video_name,
                        'prediction': 'ERROR',
                        'confidence': '0',
                        'real_prob': '0',
                        'fake_prob': '0',
                        'status': 'failed'
                    }
                    results.append(row)
                    writer.writerow(row)
                    csvfile.flush()
                    tqdm.write(f"❌ Failed to process {video_name}")
                
                # Update progress bar
                pbar.update(1)
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"📊 BATCH RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'File':<30} {'Prediction':<10} {'Confidence':<12} {'Real%':<8} {'Fake%':<8}")
    print("-" * 70)
    
    for r in results:
        if r['status'] == 'success':
            print(f"{r['file']:<30} {r['prediction']:<10} {r['confidence']:<11}% {r['real_prob']:<7}% {r['fake_prob']:<7}%")
        else:
            print(f"{r['file']:<30} {'ERROR':<10} {'-':<11}  {'-':<7}  {'-':<7}")
    
    # Statistics
    successful = [r for r in results if r['status'] == 'success']
    fake_count = sum(1 for r in successful if r['prediction'] == 'FAKE')
    real_count = sum(1 for r in successful if r['prediction'] == 'REAL')
    avg_conf = sum(float(r['confidence']) for r in successful) / len(successful) if successful else 0
    failed_count = len(results) - len(successful)
    
    print(f"\n{'-'*70}")
    print(f"Total videos tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {failed_count}")
    print(f"Predicted FAKE: {fake_count} ({fake_count/len(successful)*100:.1f}% of successful)")
    print(f"Predicted REAL: {real_count} ({real_count/len(successful)*100:.1f}% of successful)")
    print(f"Average confidence: {avg_conf:.1f}%")
    print(f"\n✅ Results saved to: {output_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python test_folder.py <folder_path>")
        print("\nExample:")
        print('  python test_folder.py "F:/deepfake/backup/TESTING/deepfake videos"')
        sys.exit(1)
    
    folder = sys.argv[1]
    test_folder(folder)
