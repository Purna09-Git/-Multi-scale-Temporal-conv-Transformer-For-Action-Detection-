from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import cv2
import numpy as np
from PIL import Image
import math
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyCpzrLNZoPspwkjE9Hp5KkgdPeohYEJ-aI"

class ActivitySegment(BaseModel):
    """Represents a time segment with detected activity"""
    start_time: int = Field(description="Start time in seconds")
    end_time: int = Field(description="End time in seconds")
    activity_description: str = Field(description="Simple but informative phrase describing what person is doing, including context (e.g., 'person is cooking at stove', 'person is walking towards door', 'person is sitting on couch', 'person is talking on phone')")
    confidence: int = Field(description="Confidence score between 1 and 100")

class VideoAnalysisResult(BaseModel):
    """Complete video analysis with activity segments"""
    total_duration: int = Field(description="Total video duration in seconds")
    activity_segments: List[ActivitySegment] = Field(description="List of activity segments")

def analyze_video_with_gemini(image_path, video_duration, api_key=None):
    """
    Analyze a merged video frames image using Gemini to extract activities.

    Args:
        image_path (str): Path to the merged frames image
        video_duration (int): Total duration of the video in seconds
        api_key (str, optional): Gemini API key if needed

    Returns:
        dict: Dictionary containing activity segments with timestamps
    """
    # Initialize Gemini client
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    # Read the image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Create system instruction
    system_instruction = f"""You are a video activity classification system. You will be shown a grid of frames from a video.
The video is {video_duration} seconds long, and the frames are arranged sequentially from left to right, top to bottom.
Your task is to:
1. Identify distinct activities in the video
2. Classify each activity with a simple but informative phrase that includes context
3. Format: "person is [doing something] [context/location]" (e.g., "person is cooking at stove", "person is walking towards door")
4. Include relevant spatial context (towards/at/on/near) and objects when visible
5. Keep phrases concise - 4-8 words maximum
6. Provide a confidence score between 1-100 for each classification
7. Determine appropriate time segments based on when activities change

Be informative but concise - like a trained classification model with context awareness."""

    # Create the prompt
    prompt = f"""Analyze this image containing sequential video frames from a {video_duration}-second video.
The frames are arranged in order from left to right, top to bottom.

Classify the activities in each time segment using simple but informative phrases with context.
Format: "person is [doing something] [context/location/direction]"

Examples of good phrases:
- "person is cooking at stove"
- "person is walking towards door"
- "person is sitting on couch"
- "person is talking on phone"
- "person is eating at table"
- "person is looking at window"
- "person is standing near counter"

Provide:
- Start and end times for each segment
- Activity phrase with context (4-8 words)
- Confidence (1-100)

Be informative but keep it concise."""

    # Generate content with structured output
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=VideoAnalysisResult,
            temperature=0.3,
        )
    )

    # Parse the response
    result = response.parsed.model_dump()

    return result

def extract_and_merge_frames(video_path, frame_interval=6):
    """
    Extract frames from video at 1-second intervals and merge them into composite images.

    Args:
        video_path (str): Path to the MP4 video file
        frame_interval (int): Number of frames to merge into one composite image

    Returns:
        list: List of PIL Image objects (merged composite images)
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    # Extract frames at 1-second intervals
    frames = []
    current_second = 0

    while current_second < duration:
        # Set position to the frame at current_second
        frame_number = int(current_second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        current_second += 1

    cap.release()

    print(f"Extracted {len(frames)} frames (one per second)")

    # Merge frames into composite images
    merged_images = []
    num_merged = math.ceil(len(frames) / frame_interval)

    for i in range(num_merged):
        start_idx = i * frame_interval
        end_idx = min(start_idx + frame_interval, len(frames))
        batch = frames[start_idx:end_idx]

        # Create composite image
        if len(batch) > 0:
            # Get dimensions of first frame
            h, w = batch[0].shape[:2]

            # Calculate grid dimensions (try to make it as square as possible)
            cols = math.ceil(math.sqrt(len(batch)))
            rows = math.ceil(len(batch) / cols)

            # Create blank canvas
            composite = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255

            # Place frames in grid
            for idx, frame in enumerate(batch):
                row = idx // cols
                col = idx % cols
                composite[row*h:(row+1)*h, col*w:(col+1)*w] = frame

            # Convert to PIL Image
            pil_image = Image.fromarray(composite)
            merged_images.append(pil_image)

    print(f"Created {len(merged_images)} merged images")

    return merged_images

def merge_all_images(images, output_path="final_merged_image.jpg", layout="horizontal"):
    """
    Merge multiple PIL images into a single image.

    Args:
        images (list): List of PIL Image objects to merge
        output_path (str): Path where the final merged image will be saved
        layout (str): Layout direction - "horizontal", "vertical", or "grid"

    Returns:
        str: Path to the saved merged image
    """
    if not images:
        raise ValueError("No images to merge")

    num_images = len(images)

    # Get dimensions of first image (assuming all are same size)
    first_img = images[0]
    img_width, img_height = first_img.size

    if layout == "horizontal":
        # Arrange images horizontally
        total_width = img_width * num_images
        total_height = img_height

        final_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

        for i, img in enumerate(images):
            final_image.paste(img, (i * img_width, 0))

    elif layout == "vertical":
        # Arrange images vertically
        total_width = img_width
        total_height = img_height * num_images

        final_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

        for i, img in enumerate(images):
            final_image.paste(img, (0, i * img_height))

    else:  # grid layout
        # Arrange in a grid (as square as possible)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)

        total_width = img_width * cols
        total_height = img_height * rows

        final_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            final_image.paste(img, (col * img_width, row * img_height))

    # Save the final image
    final_image.save(output_path, quality=95)
    print(f"Final merged image saved to: {output_path}")

    return output_path

def get_human_activity(video_file_path):
    # Replace with your video path
    video_path = video_file_path
    frame_interval = 6

    # Step 1: Extract and merge frames into intermediate images
    merged_images = extract_and_merge_frames(video_path, frame_interval)

    # Step 2: Merge all intermediate images into one final image
    merged_image_path = merge_all_images(
        merged_images,
        output_path="final_merged_image.jpg",
        layout="vertical"  # Options: "horizontal", "vertical", "grid"
    )

    print(f"Final image saved at: {merged_image_path}")

    video_duration = 18
    analysis_result = analyze_video_with_gemini(
        image_path=merged_image_path,
        video_duration=video_duration,
        # api_key=api_key  # Uncomment if needed
    )

    return analysis_result
