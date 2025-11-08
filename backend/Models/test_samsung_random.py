"""
Test random Samsung FakeAVCeleb videos (real and fake) to evaluate current model performance.
"""
import os
import glob
import random
import csv
from datetime import datetime
from tqdm import tqdm
from predict_deployment import predict_any_video

def test_samsung_random(samsung_root, num_fake=10, num_real=10, checkpoint_path="server_checkpoints/best_model.pth"):
    """Test random Samsung videos (both real and fake)."""
    
    print(f"\n{'='*70}")
    print(f"🎯 TESTING CURRENT MODEL ON SAMSUNG DATASET")
    print(f"{'='*70}")
    print(f"Samsung Root: {samsung_root}")
    print(f"Testing: {num_fake} fake + {num_real} real videos = {num_fake + num_real} total")
    print(f"{'='*70}\n")
    
    # Find fake videos
    fake_videos = []
    fake_dirs = glob.glob(os.path.join(samsung_root, "fake", "*"))
    for fake_dir in fake_dirs:
        fake_videos.extend(glob.glob(os.path.join(fake_dir, "*.mp4")))
    
    # Find real videos
    real_videos = []
    real_dirs = glob.glob(os.path.join(samsung_root, "real", "*"))
    for real_dir in real_dirs:
        real_videos.extend(glob.glob(os.path.join(real_dir, "*.mp4")))
    
    print(f"📊 Found {len(fake_videos)} fake videos and {len(real_videos)} real videos")
    
    if len(fake_videos) < num_fake:
        print(f"⚠️ Warning: Only {len(fake_videos)} fake videos available, testing all")
        num_fake = len(fake_videos)
    
    if len(real_videos) < num_real:
        print(f"⚠️ Warning: Only {len(real_videos)} real videos available, testing all")
        num_real = len(real_videos)
    
    # Randomly sample videos
    sampled_fake = random.sample(fake_videos, num_fake)
    sampled_real = random.sample(real_videos, num_real)
    
    all_videos = [(v, "FAKE") for v in sampled_fake] + [(v, "REAL") for v in sampled_real]
    random.shuffle(all_videos)
    
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"samsung_random_test_{timestamp}.csv")
    
    print(f"\n{'='*70}")
    print(f"Testing {len(all_videos)} videos...")
    print(f"Results will be saved to: {output_file}")
    print(f"{'='*70}\n")
    
    results = []
    correct = 0
    total = 0
    
    fake_correct = 0
    fake_total = 0
    real_correct = 0
    real_total = 0
    
    # Open CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'actual_label', 'prediction', 'confidence', 'real_prob', 'fake_prob', 'correct', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Test each video with progress bar
        for video_path, actual_label in tqdm(all_videos, desc="Testing videos"):
            filename = os.path.basename(video_path)
            
            try:
                # Run prediction
                result = predict_any_video(video_path, checkpoint_path)
                
                if result is not None:
                    prediction = result['prediction']
                    confidence = result['confidence'] / 100  # Convert to decimal
                    real_prob = result['real_prob'] / 100
                    fake_prob = result['fake_prob'] / 100
                    
                    # Check if prediction is correct
                    is_correct = (prediction == actual_label)
                    
                    if is_correct:
                        correct += 1
                        if actual_label == "FAKE":
                            fake_correct += 1
                        else:
                            real_correct += 1
                    
                    total += 1
                    if actual_label == "FAKE":
                        fake_total += 1
                    else:
                        real_total += 1
                    
                    # Write to CSV
                    writer.writerow({
                        'file': filename,
                        'actual_label': actual_label,
                        'prediction': prediction,
                        'confidence': f"{confidence:.2%}",
                        'real_prob': f"{real_prob:.4f}",
                        'fake_prob': f"{fake_prob:.4f}",
                        'correct': '✓' if is_correct else '✗',
                        'status': 'success'
                    })
                    
                    results.append({
                        'file': filename,
                        'actual': actual_label,
                        'predicted': prediction,
                        'correct': is_correct,
                        'confidence': confidence,
                        'real_prob': real_prob,
                        'fake_prob': fake_prob
                    })
                    
                else:
                    writer.writerow({
                        'file': filename,
                        'actual_label': actual_label,
                        'prediction': 'ERROR',
                        'confidence': 'N/A',
                        'real_prob': 'N/A',
                        'fake_prob': 'N/A',
                        'correct': '✗',
                        'status': 'prediction failed'
                    })
                    
            except Exception as e:
                writer.writerow({
                    'file': filename,
                    'actual_label': actual_label,
                    'prediction': 'ERROR',
                    'confidence': 'N/A',
                    'real_prob': 'N/A',
                    'fake_prob': 'N/A',
                    'correct': '✗',
                    'status': str(e)
                })
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 SAMSUNG DATASET TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total tested: {total} videos")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Incorrect: {total - correct} ({(total-correct)/total*100:.1f}%)")
    print(f"\n📈 Breakdown by Category:")
    if fake_total > 0:
        print(f"  FAKE videos: {fake_correct}/{fake_total} correct ({fake_correct/fake_total*100:.1f}%)")
    if real_total > 0:
        print(f"  REAL videos: {real_correct}/{real_total} correct ({real_correct/real_total*100:.1f}%)")
    print(f"\n💾 Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Show some example predictions
    print("📋 Sample Results:")
    print(f"{'File':<40} {'Actual':<8} {'Predicted':<8} {'Confidence':<12} {'Result':<8}")
    print("-" * 80)
    for r in results[:10]:  # Show first 10
        result_icon = "✓" if r['correct'] else "✗"
        print(f"{r['file']:<40} {r['actual']:<8} {r['predicted']:<8} {r['confidence']:<12.2%} {result_icon:<8}")
    
    if len(results) > 10:
        print(f"... and {len(results) - 10} more (see CSV file for full results)")
    print()
    
    return results


if __name__ == "__main__":
    samsung_root = r"F:\deepfake\backup\SAMSUNG\fakeavceleb"
    
    # Test 20 fake and 10 real videos randomly
    results = test_samsung_random(
        samsung_root=samsung_root,
        num_fake=20,
        num_real=10,
        checkpoint_path="server_checkpoints/best_model.pth"
    )
