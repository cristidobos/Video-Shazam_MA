import cv2
import numpy as np
import matplotlib.pyplot as plt
import VideoReader


def crop_video(video):
    corners = test_shit(video)
    # Get the video properties
    fps = video.fps
    height, width, channels = video.frames[0].shape

    # Create a list to store the cropped frames
    cropped_frames = []

    # Loop over each frame of the video
    for frame in video.frames:
        # Crop the frame using the specified corners
        cropped_frame = crop_image(frame, corners)

        # Append the cropped frame to the list
        cropped_frames.append(cropped_frame)

    # Create a new video object with the cropped frames
    cropped_video = type(video)(fps=fps, frames=np.array(cropped_frames))

    return cropped_video


def play_video(video):
    # Create a window to display the video
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

    # Loop over each frame of the video
    for frame in video.frames:
        # Display the frame
        cv2.imshow("Video Player", frame)

        # Wait for a key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Destroy the window
    cv2.destroyAllWindows()

def adjustContrast(frame):
    # Apply gaussian blur to remove noise
    blur = cv2.GaussianBlur(frame, (5, 5), 1)

    # Convert image to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Apply histogram equalization for contrast enhancement
    # image_equalized = cv2.equalizeHist(gray)

    return gray


# ----------------testing-------------------------------------------
def compute_frame_abs_difference(prev_frame, curr_frame):
    # Convert frames to grayscale
    # prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute the gradient between the pixels of the two frames
    gradient = cv2.absdiff(curr_frame, prev_frame)

    return gradient


def test_shit(video):
    # old_gradient = compute_frame_gradient(adjustContrast(video.frame[0]), adjustContrast(video.frame[1]))
    final_gradient = np.zeros((len(video.frames[0]), len(video.frames[0][0])))

    count = 0
    for i in range(1, len(video.frames)):
        frame1 = video.frames[i - 1]
        frame2 = video.frames[i]


        frame1 = adjustContrast(frame1)
        frame2 = adjustContrast(frame2)
        if i == 1:
            plt.imshow(frame1, cmap='gray')
            plt.show()

        gradient = compute_frame_abs_difference(frame1, frame2)
        final_gradient += gradient
        count += 1

    final_gradient *= 1 / count
    plt.imshow(gradient, cmap='gray')
    plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    grad_closing = cv2.morphologyEx(final_gradient, cv2.MORPH_CLOSE, kernel, iterations=1)
    plt.imshow(grad_closing, cmap='gray')
    plt.show()

    normalize_grad = grad_closing / np.max(grad_closing) * 255
    plt.imshow(normalize_grad, cmap='gray')
    plt.show()
    normalize_grad = remove_thin_lines(normalize_grad)

    # filtered_grad = np.where(normalize_grad < 100, 0, grad_closing)
    img_uint8 = cv2.convertScaleAbs(normalize_grad)
    plt.imshow(img_uint8, cmap='gray')
    plt.show()

    _, filtered_grad = cv2.threshold(img_uint8, 100, 255, cv2.THRESH_BINARY)
    plt.imshow(filtered_grad, cmap='gray')
    plt.show()

    # modified_grad = remove_thin_lines(filtered_grad)
    # plt.imshow(modified_grad, cmap='gray')
    # plt.show()
    # filtered_grad = modified_grad

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(filtered_grad, kernel, iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=1)
    filtered_grad = dilated
    plt.imshow(filtered_grad, cmap='gray')
    plt.show()

    corners = find_rectangle_corners(filtered_grad)
    # print(corners)
    # cropped_image = crop_image(filtered_grad, corners)
    # plt.imshow(cropped_image, cmap='gray')
    # plt.show()

    return corners


def find_rectangle_corners(image):
    # Find the coordinates of all non-zero pixels in the binary image
    coords = np.column_stack(np.where(image > 0))

    # Find the top-left and bottom-right corners of the rectangle
    tl_corner = coords.min(axis=0)
    br_corner = coords.max(axis=0)

    # Find the top-right and bottom-left corners of the rectangle
    tr_corner = np.array([br_corner[0], tl_corner[1]])
    bl_corner = np.array([tl_corner[0], br_corner[1]])

    # Return the coordinates of all four corners of the rectangle
    return [tl_corner.tolist(), tr_corner.tolist(), br_corner.tolist(), bl_corner.tolist()]


def crop_image(image, corners, padding=0.01):
    # Convert the corner coordinates to integers
    corners = np.array(corners, dtype=np.int32)

    # Find the minimum and maximum x and y coordinates of the corners
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    # Compute the width and height of the bounding box
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Add padding to the width and height
    pad_x = int(padding * width)
    pad_y = int(padding * height)

    # Adjust the coordinates of the bounding box by the padding
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image.shape[0] - 1, x_max + pad_x)
    y_max = min(image.shape[1] - 1, y_max + pad_y)

    # Crop the image to the adjusted bounding box
    cropped_image = image[x_min:x_max + 1, y_min:y_max + 1]

    return cropped_image


def remove_thin_lines(image):
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(erode, kernel, iterations=1)
    return dilated
