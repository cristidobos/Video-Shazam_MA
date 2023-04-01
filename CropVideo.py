import cv2
import numpy as np
import matplotlib.pyplot as plt

from Video import Video


def crop_video(video):
    # The list of frames
    frames = video.frames

    # Iterate through each frame
    for i in range(len(frames)):
        frame = frames[i]
        # Adjust contrast of frame
        adjusted = adjustContrast(frame)

        # Find countours of objects within frame
        contours = findCountours(adjusted)

        # Crop the frame given the largest contour
        cropped = cropFrame(contours, frame)

        # Replace with cropped
        frames[i] = cropped
    return video

def adjustContrast(frame):
    # Apply gaussian blur to remove noise
    blur = cv2.GaussianBlur(frame, (3, 3), 0)

    # Convert image to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for contrast enhancement
    #image_equalized = cv2.equalizeHist(gray)

    return gray

def findCountours(image):
    # Perform adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    #thresh = cv2.Canny(image, 100, 200)
    # Find contours present in image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return cnts

def cropFrame(contours, original):
    # Select the largest contour
    display = sorted(contours, key = cv2.contourArea, reverse=True)[0]

    # Find bounding box of object
    x, y, w, h = cv2.boundingRect(display)

    #cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # Crop the frame around bounding box
    cropped = original[y : y + h, x : x + w, :]

    #return original
    return cropped



