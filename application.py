import video_query
# import CropVideo
# from VideoReader import Video
import FileReader


def main():
    print("Please provide the path of the video query:")
    print("example: dataset/Asteroid_Discovery")

    query_path = str(input())
    database = "database/videos.sqlite"

    # Store query as a video object
    video = FileReader.VideoReader(query_path + ".mp4")
    audio = FileReader.AudioReader(query_path + ".wav")

    # Crop the video only to contain the video contents
    # cropped_video = CropVideo.crop_video(video)

    print("Parsed query")
    # Get descriptor for query video
    descriptor = video_query.get_query_descriptor(video, audio)
    print("Got descriptor")
    # Find best match by comparing with all database entries
    name, score, frame = video_query.find_best_match(descriptor, database, audio.sample_rate)

    print("Best match was found for video {} with a score of {} at frame {}".format(name, score, frame))


if __name__ == '__main__':
    main()
