"""
Extract frames from Video Database for training
This script processes videos and extracts labeled frames for drowsy/alert states
"""

import cv2
import os
import numpy as np
from pathlib import Path

def extract_frames_from_videos():
    """
    Extract frames from Video Database videos
    Saves frames in a structured format for training
    """
    video_dir = Path("Video Database")
    output_dir = Path("extracted_video_frames")
    
    # Create output directories
    (output_dir / "drowsy").mkdir(parents=True, exist_ok=True)
    (output_dir / "alert").mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob("*.avi"))
    
    print(f"📹 Found {len(video_files)} video files")
    
    for video_path in video_files:
        print(f"\n🎬 Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ❌ Cannot open {video_path.name}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  Total frames: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        saved_count = 0
        
        # Sample every 30 frames (1 frame per second at 30fps)
        sample_rate = max(1, fps // 1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Save frame
                # Note: Manual labeling needed - for now save all as "alert"
                # You can modify this to detect drowsy frames automatically
                filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
                output_path = output_dir / "alert" / filename
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
        
        cap.release()
        print(f"  ✅ Saved {saved_count} frames")
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 Frames saved in: {output_dir}")
    print(f"⚠️  Note: Manual labeling may be needed to separate drowsy/alert frames")

if __name__ == "__main__":
    extract_frames_from_videos()
