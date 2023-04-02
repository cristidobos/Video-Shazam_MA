import cv2
import numpy as np
import pyaudio
import scipy.io.wavfile as wav

"""
Takes as an input the path to a video, and returns its frames and fps

input: path of the video
output: fps and array of frames 

"""


class VideoReader:

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
            # Read video frame by frame
            ret, frame = cap.read()

            if ret == True:
                # Append frame to the list of frames
                frames.append(frame)
                # cv2.imshow("Frame", frame)
            else:
                break

            # key = cv2.waitKey(30)
            # # if key q is pressed then break
            # if key == 113:
            #     break
        # Close video file
        cap.release()
        cv2.destroyAllWindows()

        return fps, frames

    def display_video(self):
        for frame in self.frames:
            cv2.imshow('Video', frame)

            if cv2.waitKey(self.fps) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


class AudioReader:
    def __init__(self, path):
        self.path = path
        # Read audio
        self.audio_data, self.sample_rate = self.readAudio(path)

    def readAudio(self, path):
        # audiopath = '../resource/lib/publicdata/Videos/BlackKnight.wav'
        sample_rate, audio_data = wav.read(path)
        audio_data = audio_data.astype('float32') / 32767.0

        return audio_data, sample_rate

    def play_audio(self):
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open audio stream
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sample_rate,
                        output=True)

        # Play audio
        stream.write(self.audio_data)

        # Close audio stream and PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()


path = "training/video1.mp4"
if __name__ == '__main__':
    video = VideoReader(path)

    print(len(video.frames))
