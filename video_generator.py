import cv2
import os

# Paths
OUTPUT_DIR = r'Frame_Interpolation\predicted_frames'  # Directory with predicted frames
VIDEO_PATH = r'Frame_Interpolation\interpolated_video.mp4'  # Output video file

# Parameters
FRAME_RATE = 1  # Frames per second (adjust as needed)

# Get list of image files in sorted order
image_files = sorted(
    [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')],
    key=lambda x: int(x.split('_')[1].split('.')[0])  # Assumes filenames are like predicted_1.png, predicted_2.png
)

# Read the first image to get frame size
first_frame = cv2.imread(os.path.join(OUTPUT_DIR, image_files[0]))
height, width, layers = first_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FRAME_RATE, (width, height))

# Write frames to the video
for image_file in image_files:
    frame_path = os.path.join(OUTPUT_DIR, image_file)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

# Release the VideoWriter
video_writer.release()

print(f"Video created successfully at {VIDEO_PATH}")
