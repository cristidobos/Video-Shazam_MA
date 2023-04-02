import sqlite3
import sys
import sqlite3 as sqlite
import cv2
import numpy as np
import database
import video_features
from database import adapt_array, convert_array
from video_features import *

def find_best_match(video_descriptor, database_path):
    # Connect to the database
    connection = None
    try:
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        connection = sqlite3.connect(database_path, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
    except ConnectionRefusedError as e:
        print(e)

    # Create cursor object
    cursor = connection.cursor()

    # Execute query
    cursor.execute('SELECT * FROM Video')

    # Fetch all entries in database
    videos = cursor.fetchall()

    video_name = ''
    f = None
    best_score = None
    # Iterate over all videos
    for (_, name, mfcc, audio, colhist, tempdiff, chdiff) in videos:
        if len(video_descriptor['mfcc']) > len(mfcc):
            continue
        # Compare only based on color histogram
        frame, score = sliding_window(colhist, video_descriptor['colhist'], euclidean_norm_mean)

        if best_score is None:
            best_score = score
            video_name = name
            f = frame
        else:
            if score < best_score:
                best_score = score
                f = frame
                video_name = name

    return video_name, best_score, f


def get_query_descriptor(video):
    if len(video.frames) == 0:
        raise Exception("Video length is 0")

    colorhist = []
    tempdiff = []
    audiopowers = []
    mfccs = []
    colorhistdiffs = []
    prev_frame = None
    prev_colorhist = None

    for frame in video.frames:
        hist = video_features.colorhist(frame)
        colorhist.append(hist)
        tempdiff.append(temporal_diff(prev_frame, frame, 10))
        colorhistdiffs.append(colorhist_diff(prev_colorhist, hist))
        prev_colorhist = hist

    descriptor = {}
    descriptor['mfcc'] = np.array(mfccs[:-1])
    descriptor['audio'] = np.array(audiopowers)
    descriptor['colhist'] = np.array(colorhist)
    descriptor['tempdiff'] = np.array(tempdiff)
    descriptor['chdiff'] = np.array(colorhistdiffs)

    return descriptor

def sliding_window(x, w, compare_func):
    """ Slide window w over signal x.

        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    frame = -1 # init frame
    wl = len(w)
    minimum = sys.maxsize
    for i in range(len(x) - wl):
        diff = compare_func(w, x[i:(i+wl)])
        if diff < minimum:
            minimum = diff
            frame = i
    return frame, minimum

def euclidean_norm_mean(x, y):
    x = np.mean(x, axis=0)
    y_m = 0
    if len(y) > 0:
        y_m = np.mean(y, axis=0)
    return np.linalg.norm(x-y)

def euclidean_norm(x, y):
    return np.linalg.norm(x-y)

