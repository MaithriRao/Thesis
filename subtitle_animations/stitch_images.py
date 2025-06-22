import os
import re
import subprocess

frames_path = "/home/maithri/Documents/SignDataResults/INFERENCE-ASL/rawFrames/SGlWxxp-IBA"

start_time = "00:00:24,312"
end_time = "00:00:27,000"

fps = 30 
nr_of_images = 8

def timestamp_to_frame(timestamp, fps):
    hh, mm, ss, ms = map(float, re.split(r'[:,]', timestamp))
    total_seconds = hh * 3600 + mm * 60 + ss + ms / 1000
    frame_number = round(total_seconds * fps)
    return frame_number

start_time = timestamp_to_frame(start_time, fps)
end_time = timestamp_to_frame(end_time, fps)

print(f"From frame {start_time} to frame {end_time}")


number_to_filename = {}

for filename in os.listdir(frames_path):
    if filename == 'fps.txt':
        continue
    filename_no_ext = os.path.splitext(filename)[0]
    match = re.match(r'.*(\d{7})$', filename_no_ext)
    if match is None:
        print(f"Failed to parse frame number from {filename_no_ext}")
        continue
    (frame_number,) = match.groups()
    frame_number = int(frame_number)
    number_to_filename[frame_number] = filename


left_edge = start_time - 0.5
right_edge = end_time + 0.5

denominator = nr_of_images * 2

chosen_frames = []
for numerator in range(1, denominator, 2):
    # print(numerator, denominator)
    relative_position = (right_edge - left_edge) * numerator / denominator
    position = relative_position + left_edge
    position = round(position)
    # print(position)
    chosen_frames.append(position)

print(chosen_frames)

chosen_files = [number_to_filename[frame_number] for frame_number in chosen_frames]

print(chosen_files)

chosen_file_paths = [os.path.join(frames_path, file_name) for file_name in chosen_files]

subprocess.run(["magick"] + chosen_file_paths + ["-crop", "820x720+320+0>!", "+append", "video.png"], check=True)
