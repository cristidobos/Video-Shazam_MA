#!/usr/bin/env python
import subprocess
import re
from fractions import Fraction
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import glob
from scipy.io.wavfile import read
from mfcc import *
from video_features import *
from database import *

# Processing of videos
    def process_videos(video_list, connection):
    total = len(video_list)
    progress_count = 0
    for video in video_list:

        progress_count += 1
        print('processing: ', video, ' (', progress_count, ' of ', total, ')')
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            raise Exception("No video file found at the path specified")

        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        # get corresponding audio file
        filename, fileExtension = os.path.splitext(video)
        audio = filename + '.wav'
        fs, wav_data = read(audio)

        colorhists = []
        sum_of_differences = []
        audio_powers = []
        mfccs = []
        colorhist_diffs = []

        prev_colorhist = None
        prev_frame = None
        frame_nbr = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            audio_frame = frame_to_audio(frame_nbr, frame_rate, fs, wav_data)

            # check if audio frame is long enough for mfcc transformation
            if len(audio_frame) >= int(0.01 * fs):
                power = np.mean(audio_frame ** 2)
                audio_powers.append(power)
                ceps, mspec, spec = extract_mfcc(audio_frame, fs)
                mfccs.append(ceps)

            # calculate sum of differences
            if not prev_frame is None:
                tdiv = temporal_diff(prev_frame, frame, 10)
                # diff = np.absolute(prev_frame - frame)
                # sum = np.sum(diff.flatten()) / (diff.shape[0]*diff.shape[1]*diff.shape[2])
                sum_of_differences.append(tdiv)
            color_hist = colorhist(frame)
            colorhists.append(color_hist)
            if not prev_colorhist is None:
                ch_diff = colorhist_diff(prev_colorhist, color_hist)
                colorhist_diffs.append(ch_diff)
            prev_colorhist = color_hist
            prev_frame = frame
            frame_nbr += 1
        print('end:', frame_nbr)

        # prepare descriptor for database
        # mfccs = descr['mfcc'] # Nx13 np array (or however many mfcc coefficients there are)
        # audio = descr['audio'] # Nx1 np array
        # colhist = descr['colhist'] # Nx3x256 np array
        # tempdif = descr['tempdiff'] # Nx1 np array
        descr = {}
        descr['mfcc'] = np.array(mfccs[:-1])
        descr['audio'] = np.array(audio_powers)
        descr['colhist'] = np.array(colorhists)
        descr['tempdiff'] = np.array(sum_of_differences)
        descr['chdiff'] = np.array(colorhist_diffs)
        video_name = video
        video_name.replace('dataset/', '')
        add_video_descriptor(progress_count, video, descr, connection)
        print('added ' + video + ' to database')
    connection.commit()





