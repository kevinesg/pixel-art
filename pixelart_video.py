from functions import pixel_art
from functions import write_video
import cv2
import os


VIDEO_FILE = '''video file path'''
vid = cv2.VideoCapture(VIDEO_FILE)

# Initialize the list of frames
frames = []
i = 1   # Counter
while True:
    grabbed, frame = vid.read()
    print(f'[INFO] Processing frame #{i}...')

    if grabbed:
        # Save the frame (pixel_art function requires the frame in a file format)
        cv2.imwrite('frame.jpg', frame)
        # Generate the pixel art of the frame
        pixel_frame = pixel_art('frame.jpg')
        frames.append(pixel_frame)

    i += 1

vid.release()
os.remove('frame.jpg')
 
write_video('''new video filename''', frames, fps=24)