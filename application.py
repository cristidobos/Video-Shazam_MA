from Video import Video
import video_query
import CropVideo

def main():
    print("Please provide the path of the video query:")
    query_path = str(input())
    database = "database/videos.sqlite"

    # Store query as a video object
    video = Video(query_path)
    print("Parsed query")
    # Get descriptor for query video
    descriptor = video_query.get_query_descriptor(video)
    print("Got descriptor")
    # Find best match by comparing with all database entries
    name, score, frame = video_query.find_best_match(descriptor, database)

    print("Best match was found for video {} with a score of {} at frame {}".format(name, score, frame))



if __name__ == '__main__':
    main()
