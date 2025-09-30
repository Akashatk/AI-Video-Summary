import os
import subprocess
import glob
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Set paths
VIDEO_PATH = "input.mp4"
SEGMENT_DIR = "segments"
FRAMES_DIR = "frames"
os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def split_video(video_path, output_dir, segment_length=10):
    """Split the video into segments of specified length (in seconds)"""
    output_template = os.path.join(output_dir, "segment_%03d.mp4")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-c", "copy", "-map", "0",
        "-f", "segment", "-segment_time", str(segment_length),
        "-reset_timestamps", "1", output_template
    ])

def extract_frames(segment_path, output_dir, fps=3):
    """Extract frames at specified fps from a video segment"""
    segment_name = os.path.splitext(os.path.basename(segment_path))[0]
    segment_frame_dir = os.path.join(output_dir, segment_name)
    os.makedirs(segment_frame_dir, exist_ok=True)
    output_pattern = os.path.join(segment_frame_dir, "frame_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", segment_path, "-vf", f"fps={fps}",
        output_pattern
    ])
    return segment_frame_dir

def describe_frame(image_path):
    """Use BLIP to describe an image"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_captions():
    """Main function to generate captions for video segments"""
    # Step 1: Split the video
    print("Splitting video into 10s segments...")
    split_video(VIDEO_PATH, SEGMENT_DIR)

    # Step 2 & 3: Extract frames and run BLIP
    print("Processing segments...")
    all_captions = {}

    segment_files = sorted(glob.glob(os.path.join(SEGMENT_DIR, "*.mp4")))

    for segment_file in segment_files:
        print(f"Processing {segment_file}")
        frame_dir = extract_frames(segment_file, FRAMES_DIR, fps=3)
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

        captions = []
        for frame_file in frame_files:
            caption = describe_frame(frame_file)
            captions.append({
                "frame": os.path.basename(frame_file),
                "caption": caption
            })
            print(f"  - {os.path.basename(frame_file)}: {caption}")

        all_captions[os.path.basename(segment_file)] = captions

    # Optional: Save captions to JSON
    import json
    with open("captions_output.json", "w") as f:
        json.dump(all_captions, f, indent=2)

    print("Done! Captions saved to captions_output.json.")
