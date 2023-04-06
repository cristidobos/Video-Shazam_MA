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
        if len(video_descriptor['mfcc']) > len(mfcc) or len(video_descriptor['mfcc'][0]) != len(mfcc[0]) or len(
                video_descriptor['mfcc'][0][0]) != len(mfcc[0][0]):
            continue

        # Compare only based on color histogram
        # frame, score = find_multiple_best(colhist, video_descriptor['colhist'], euclidean_norm_mean, 1, frame_rate)

        pad1_database_element, pad2_database_element = pad_arrays(mfcc, colhist)
        database_element_feature = np.hstack((pad1_database_element, pad2_database_element))

        # I'm padding both features to the same shape so that I can stack them together
        pad1_query_element , pad2_query_element = pad_arrays(video_descriptor['mfcc'], video_descriptor['colhist'])

        query_element_feature = np.hstack((pad1_query_element, pad2_query_element))
        # frame, score = find_multiple_best(mfcc, video_descriptor['mfcc'], euclidean_norm_mean, 1, frame_rate)

        frames_and_scores = find_multiple_best(database_element_feature, query_element_feature, euclidean_norm_mean, 1,
                                               frame_rate)
        frame, score = frames_and_scores

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


def get_query_descriptor(video, audio):
    if len(video.frames) == 0:
        raise Exception("Video length is 0")

    samples_per_audio_frame = int((1 / video.fps) * audio.sample_rate)

    counter = 0
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
        if counter + samples_per_audio_frame < len(audio.audio_data):
            audio_sample = audio.audio_data[counter: counter + samples_per_audio_frame]
            ceps, mspec, spec = extract_mfcc(audio_sample, audio.sample_rate)
            mfccs.append(ceps[:-1, :])
            counter += samples_per_audio_frame

    # mfccs = ceps

    descriptor = {}
    descriptor['mfcc'] = np.array(mfccs)
    descriptor['audio'] = np.array(audiopowers)
    descriptor['colhist'] = np.array(colorhist)
    descriptor['tempdiff'] = np.array(tempdiff)
    descriptor['chdiff'] = np.array(colorhistdiffs)

    return descriptor


def sliding_window(x, w, compare_func):
    """
    Slide window w over signal x.

        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    replace_frame = np.ones(x[0].shape)
    frame = -1  # init frame
    wl = len(w)
    diffs = []
    minimum = sys.maxsize
    shift = 10
    i = 0
    while i < len(x) - wl:
        first_frame = x[i]
        last_frame = x[i + wl - 1]
        if np.array_equal(first_frame, replace_frame) or np.array_equal(last_frame, replace_frame):
            i += 1
            continue
        diff = compare_func(w, x[i:(i + wl)])
        diffs.append(diff)
        if diff < minimum:
            minimum = diff
            frame = i
        i += shift

    return frame, minimum


def find_multiple_best(x, w, compare_func, n):
    replace_frame = np.ones(x[0].shape)
    frames = np.zeros((n,))
    mins = np.zeros((n,))
    x_copy = np.array(x)

    frame, minimum = sliding_window(x_copy, w, compare_func)
    # best_matches.append((frame, minimum))
    #frames[i] = frame
    #mins[i] = minimum
    #x_copy[frame: frame + len(w)] = 1
    # x_copy = np.concatenate((x_copy[:frame, :, :], x_copy[(frame + len(w)):, :, :]))
    return frame, minimum


def euclidean_norm_mean(x, y):
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x - y)


def euclidean_norm(x, y):
    return np.linalg.norm(x - y)
