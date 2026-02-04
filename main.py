from codec import *
import cv2


# OpenCV (cv2) library is only used to read and store a file in the next two functions
# Save a list of frames as a video
def save_frames_as_video(target_directory, fps, frames):
    if not frames:
        raise ValueError("The frame list is empty.")

    # Use the shape of the first frame to get video dimensions
    height, width, _ = frames[0].shape

    # Use fourcc code for a more compatible codec (XVID in this case)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(target_directory, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()


# Convert a video to a list of frames
def video_to_frames(source_directory):
    video = cv2.VideoCapture(source_directory)
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {source_directory}")
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()
    return frames


# Main program
# Convert the video to frames
video_frames = video_to_frames("original_video.mp4")
i_frames = video_frames[::12]
p_frames = [frame for f, frame in enumerate(video_frames) if f % 12 != 0]

# Part i
# Encode frames and I-frames
encoded_errors, encoded_i_frames = encode_frames_to_compressed(video_frames, i_frames)
# Decode video from encoded error frames and I-frames
decoded_frames = decode_error_frames(encoded_errors, encoded_i_frames)
# Save the reconstructed video for the first part
save_frames_as_video("output_video_1st_part.mp4", 24, decoded_frames)

# Part ii
# Encode video using exhaustive motion compensation
motion_vectors_exhaustive = emc_video_encode(decoded_frames)

# Part iii
# Encode video using logarithmic motion compensation
motion_vectors_log, error_frames_log = log_encode(decoded_frames)
# Decode video using logarithmic motion compensation
reconstructed_frames_log = log_decode(i_frames, motion_vectors_log, error_frames_log)
# Save the reconstructed video for the third part
save_frames_as_video("output_video_3rd_part.mp4", 24, reconstructed_frames_log)

# Part iv
# Calculate the original size of the video frames
original_video_size = sum(frame.nbytes for frame in video_frames)

# Part i compression ratio
# Calculate sizes for the components
encoded_i_size = sum(len(frame) for frame in encoded_i_frames)
encoded_errors_size = sum(len(frame) for frame in encoded_errors)
total_encoded_size = encoded_i_size + encoded_errors_size
compression_ratio_first_part = original_video_size / total_encoded_size
print(f"The first part has a compression ratio of: {compression_ratio_first_part:.2f}")

# Part iii compression ratio
# Calculate sizes for the components
motion_vectors_log_size = sum(vector.nbytes for vector in motion_vectors_log)
error_frames_log_size = sum(frame.nbytes for frame in error_frames_log)
i_frames_size = sum(frame.nbytes for frame in i_frames)
total_encoded_size_log = motion_vectors_log_size + error_frames_log_size + i_frames_size
compression_ratio_third_part = original_video_size / total_encoded_size_log
print(f"The third part has a compression ratio of: {compression_ratio_third_part:.2f}")
