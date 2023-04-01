import cv2
import numpy as np
import matplotlib.pyplot as plt

import CropVideo


class Video:

    def __init__(self, path):
        self.path = path
        # Read videos
        self.fps, self.frames = self.readVideo(path)

    def readVideo(self, path):
        cap = cv2.VideoCapture(path)

        # Get video frames per second
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            raise Exception("Not read file")
        # Initialize frames as an empty list
        frames = []
        for i in range(total_frames):
            # Read video frame by frame
            ret, frame = cap.read()

            if ret == True:
                # Append frame to the list of frames
                frames.append(frame)
            else:
                break

        # Close video file
        cap.release()

        return fps, frames

