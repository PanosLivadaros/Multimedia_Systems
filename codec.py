from tqdm import tqdm
import imageio
import numpy
import io


# Calculate the Sum of Absolute Differences (SAD) between two blocks
def sum_of_ad(block_1, block_2):
    return numpy.sum(numpy.abs(block_1 - block_2))


# Perform Exhaustive Motion Compensation to calculate the motion vectors
def emc(current, reference, b_size, radius):
    height, width, _ = current.shape
    motion_vectors = numpy.zeros((height // b_size, width // b_size, 2), dtype=numpy.int8)

    for i in range(0, height - b_size + 1, b_size):
        for j in range(0, width - b_size + 1, b_size):
            current_b = current[i:i + b_size, j:j + b_size]
            min_sad, best = float('inf'), (0, 0)
            for m in range(-radius, radius + 1):
                for n in range(-radius, radius + 1):
                    reference_i, reference_j = i + m, j + n
                    if 0 <= reference_i < height - b_size and 0 <= reference_j < width - b_size:
                        reference_b = reference[reference_i:reference_i + b_size, reference_j:reference_j + b_size]
                        sad = sum_of_ad(current_b, reference_b)
                        if sad < min_sad:
                            min_sad, best = sad, (m, n)
            motion_vectors[i // b_size, j // b_size] = best
    return motion_vectors


# Perform logarithmic search to calculate the motion vectors
def log_search(current, reference, b_size, radius):
    height, width, _ = current.shape
    motion_vectors = numpy.zeros((height // b_size, width // b_size, 2), dtype=numpy.int8)

    for i in range(0, height - b_size + 1, b_size):
        for j in range(0, width - b_size + 1, b_size):
            current_b = current[i:i + b_size, j:j + b_size]
            min_sad, best = float('inf'), (0, 0)
            step = radius // 2
            while step >= 1:
                for m in [-step, 0, step]:
                    for n in [-step, 0, step]:
                        reference_m, reference_n = i + best[0] + m, j + best[1] + n
                        if 0 <= reference_m < height - b_size and 0 <= reference_n < width - b_size:
                            reference_b = reference[reference_m:reference_m + b_size, reference_n:reference_n + b_size]
                            sad = sum_of_ad(current_b, reference_b)
                            if sad < min_sad:
                                min_sad, best = sad, (best[0] + m, best[1] + n)
                step //= 2
            motion_vectors[i // b_size, j // b_size] = best
    return motion_vectors


# Generate a predicted frame given the reference frame and the motion vectors
def create_predicted_frame(reference, motion_vectors, b_size):
    height, width, _ = reference.shape
    predicted = numpy.zeros_like(reference)

    for i in range(0, height - b_size + 1, b_size):
        for j in range(0, width - b_size + 1, b_size):
            motion_vector = motion_vectors[i // b_size, j // b_size]
            reference_i, reference_j = i + motion_vector[0], j + motion_vector[1]
            if 0 <= reference_i < height - b_size and 0 <= reference_j < width - b_size:
                predicted[i:i + b_size, j:j + b_size] = reference[reference_i:reference_i + b_size, reference_j:reference_j + b_size]
    return predicted


# Encode frames into the compressed error frames and the I-frames
def encode_frames_to_compressed(frames, i_frames):
    error_frames = [frames[i] - frames[i - 1] for i in range(1, len(frames)) if i % 12 != 0]
    encoded_error_frames = png_encode_frames(error_frames)
    encoded_i_frames = png_encode_frames(i_frames)
    return encoded_error_frames, encoded_i_frames


# Decode the encoded error frames and the I-frames to reconstruct the original frames
def decode_error_frames(encoded_error_frames, encoded_i_frames):
    error_frames = decode_encoded_frames(encoded_error_frames)
    i_frames = decode_encoded_frames(encoded_i_frames)
    count = 0
    frames = []
    for i, i_frame in enumerate(i_frames):
        frames.append(i_frame)
        for _ in range(11):
            if count < len(error_frames):
                frame = frames[-1] + error_frames[count]
                frames.append(frame)
                count += 1
    return frames


# Encode the frames using PNG compression
def png_encode_frames(frames):
    def png_encode_single_frame(frm):
        bytes_data = io.BytesIO()
        imageio.imwrite(bytes_data, frm, format="png")
        data = bytes_data.getvalue()
        bytes_data.close()
        return data

    encoded_frames = []
    for frame in tqdm(frames, desc="Encoding Frames", unit="frm"):
        try:
            encoded_frames.append(png_encode_single_frame(frame))
        except Exception as e:
            print(f"An error occurred while encoding a frame: {e}")
    return encoded_frames


# Decode the encoded frames
def decode_encoded_frames(encoded):
    def decode_single_encoded_frame(encoded_frame):
        bytes_data = io.BytesIO(encoded_frame)
        decoded_frame = imageio.imread(bytes_data, format="png")
        bytes_data.close()
        return decoded_frame

    decoded_frames = []
    for encoded in tqdm(encoded, desc="Decoding Frames", unit="frame"):
        try:
            decoded_frames.append(decode_single_encoded_frame(encoded))
        except Exception as e:
            print(f"An error occurred while decoding a frame: {e}")
    return decoded_frames


# Generate the motion vectors and the error frame using logarithmic search
def log_frame_encode(current, reference, b_size, domain):
    motion_vectors = log_search(current, reference, b_size, domain)
    predicted = create_predicted_frame(reference, motion_vectors, b_size)
    error = numpy.subtract(current.astype(numpy.uint8), predicted.astype(numpy.uint8))
    return motion_vectors, error


# Encode the video frames into the motion vectors and the error frames using logarithmic search
def log_encode(frames):
    vectors, error = [], []
    for i in tqdm(range(1, len(frames)), desc="Encoding Video with Logarithmic Search", unit="frame"):
        if i % 12 != 0:
            single_vector, single_error_frame = log_frame_encode(frames[i], frames[i - 1], 64, 4)
            vectors.append(single_vector)
            error.append(single_error_frame)
    return vectors, error


# Encode the video frames into the motion vectors using exhaustive motion compensation
def emc_video_encode(frames):
    vectors = []
    for i in tqdm(range(1, len(frames)), desc="Encoding Video with Exhaustive Motion Compensation", unit="frame"):
        if i % 12 != 0:
            single_vector = emc(frames[i], frames[i - 1], 16, 8)
            vectors.append(single_vector)
    return vectors


# Reconstruct the video frames from the I-frames, the motion vectors, and the error frames
def log_decode(i_frames, vectors, error):
    error_index, i_frame_index = 0, 0
    reconstructed = []
    for i in tqdm(range(len(i_frames) + len(vectors)), desc="Decoding Video with Logarithmic Search", unit="frame"):
        if i % 12 == 0:
            reconstructed.append(i_frames[i_frame_index])
            i_frame_index += 1
        else:
            if error_index < len(vectors):
                reference = reconstructed[-1]
                single_motion_vector = vectors[error_index]
                single_error_frame = error[error_index]
                single_reconstructed_frame = create_predicted_frame(reference, single_motion_vector, 64)
                single_reconstructed_frame = (single_reconstructed_frame + single_error_frame).clip(0, 255).astype(numpy.uint8)
                reconstructed.append(single_reconstructed_frame)
                error_index += 1
    return reconstructed
