import subprocess
import re
from fractions import Fraction
import cv2
from mfcc import *

def colorhist(im):
    chans = cv2.split(im)
    color_hist = np.zeros((256,len(chans)))
    for i in range(len(chans)):
        color_hist[:,i] = np.histogram(chans[i], bins=np.arange(256+1))[0]/float((chans[i].shape[0]*chans[i].shape[1]))
    return color_hist

# From video_tools.py ### Not sure
def video_info(video, util):
    cmd = util + ' -show_streams ' + video
    process = subprocess.Popen(cmd.split(), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out.decode('utf-8')

def get_frame_rate(video, util='ffprobe'):
    info = video_info(video, util)
    pattern = 'codec_type\=video.*?avg_frame_rate\=(\d+[\/\d.]*|\d)'
    result = re.search(pattern, info, re.DOTALL).group(1)
    return float(Fraction(result))

def get_frame_count(video, util='ffprobe'):
    info = video_info(video, util)
    pattern = 'codec_type\=video.*?nb_frames\=([0-9]+)'
    result = re.search(pattern, info, re.DOTALL)
    return int(result.group(1))

def get_frame_count_audio(video, util='ffprobe'):
    info = video_info(video, util)
    pattern = 'codec_type\=audio.*?nb_frames\=([0-9]+)'
    result = re.search(pattern, info, re.DOTALL)
    return int(result.group(1))

def frame_to_audio(frame_nbr, frame_rate, fs, audio):
    start_index = int(frame_nbr / frame_rate * fs)
    end_index = int((frame_nbr+1) / frame_rate * fs)
    return audio[start_index:end_index]

def temporal_diff(frame1, frame2, threshold=50):
    if frame1 is None or frame2 is None:
        return None
    diff = np.abs(frame1.astype('int16') - frame2.astype('int16'))
    diff_t = diff > threshold
    return np.sum(diff_t)


def colorhist_diff(hist1, hist2):
    if hist1 is None or hist2 is None:
        return None
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)