import cv2
import numpy as np
import matplotlib.pyplot as plt

import CropVideo


class Video:

    def __init__(self, path = None, fps=None, frames=None):
        self.path = path
        if fps is None or frames is None:
            # Read videos
            self.fps, self.frames = self.readVideo(path)
        else:
            self.fps = fps
            self.frames = frames

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
        #for i in range(5): # only for testing
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

    def testing(self):
        frames = self.frames
        f = []
        # Iterate through each frame
        for i in range(len(frames)):
            frame = frames[i]
            # Adjust contrast of frame
            # Apply gaussian blur to remove noise
            blur = cv2.GaussianBlur(frame, (3, 3), 0)

            # Convert image to grayscale
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization for contrast enhancement
            adjusted = cv2.equalizeHist(gray)

            # Find countours of objects within frame
            # Perform adaptive thresholding to get a binary image
            #thresh = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
            thresh = cv2.Canny(adjusted, 100, 200)
            # Find contours present in image
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)
            f.append(frame)

        plt.imshow(f[0])
        plt.show()


path = "training/video5.mp4"
if __name__ == '__main__':
    video = Video(path)
    cropped = CropVideo.crop_video(video)
    # CropVideo.test_shit(video)
    # cropped2 = CropVideo.crop_video(cropped)
    # CropVideo.play_video(cropped)
    # for i in range(10):
    #     plt.imshow(cropped.frames[i])
    #     plt.show()

