import numpy as np
import cv2
from colorthief import ColorThief


# Generates a pixel art of an individual frame/image
def pixel_art(frame, num_clusters=20, scale=8):
    '''
    frame: The individual frame of a video file (frames list accessed in a for loop)
    num_clusters: The total number of different colors in each frame
    scale: Each "square" in an output frame has scale x scale pixels (scale=8 means 8x8 pixels)

    '''

    img = cv2.imread(frame)
    ct = ColorThief(frame)
    # Get the cluster points
    palette = ct.get_palette(color_count=num_clusters)
    # Reverse the channels from RGB to BGR (OpenCV uses BGR format)
    new_palette = []
    for point in palette:
        new_point = [0, 0, 0]
        new_point[0] = point[2]
        new_point[1] = point[1]
        new_point[2] = point[0]
        new_palette.append(new_point)
    palette = new_palette

    # Get the rescaled frame dimensions depending on the scale
    frame_resized = cv2.resize(img, [int(img.shape[1] / scale), int(img.shape[0] / scale)])
    # Create a blank canvas where we will paint the pixel art
    canvas = np.zeros((img.shape))
    
    for y in range(frame_resized.shape[0]):
        for x in range(frame_resized.shape[1]):
            pixel = frame_resized[y, x]
            d = 10000   # Arbitrarily large initial distance (will be replaced immediately)
            for point in palette:
                # Calculate Euclidean distance
                distance = np.sqrt(
                    (point[0] - pixel[0]) ** 2 + (point[1]- pixel[1]) ** 2 + (point[2]- pixel[2]) ** 2
                )

                # Choose the nearest cluster point
                if distance < d:
                    d = distance
                    cluster = point

            # Paint the canvas with the correct cluster point        
            ix = int(scale * x)
            iy = int(scale * y)
            ix_ = int(scale * (x + 1))
            iy_ = int(scale * (y + 1))
            canvas[iy:iy_, ix:ix_] = cluster

    # Convert float to int because opencv only accepts int values
    canvas = canvas.astype(np.uint8)

    return canvas


# Compiles a list of frames/images into a video file with a specified fps
def write_video(file_path, frames, fps):
    '''
    file_path: The file path or simply file name of the mp4 file that will be created
    frames: the list of frames
    fps: frames per second of the new video file

    '''

    h, w, _ = frames[0].shape
    # fourcc: four character code; required by the VideoWriter function
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    # Compile the frames into a video file
    for i, frame in enumerate(frames):
        print(f'[INFO] Writing frame #{i + 1}')
        writer.write(frame)
    
    writer.release()