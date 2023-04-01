import CropVideo
from Video import Video


def analyse_video(path):
    # Fetch query video from path
    query_video = Video(path)

    # Crop the video to the region of interest
    cropped_video = CropVideo.crop_video(query_video)


