import cv2
import pafy
import numpy as np
import os
from raft import Raft
import re
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import tqdm

from concurrent.futures import Executor, Future, wait, FIRST_COMPLETED
from typing import Callable, Iterable, Iterator, TypeVar
from typing_extensions import TypeVar

STATE_ESTIMATING = 1
STATE_COMPLETED = 2
STATE_FAILED = 3

# Initialize model
model_path ='models/raft_things_iter20_480x640.onnx'
flow_estimator = Raft(model_path)

db_path = '/ds/videos/opticalflow-BOBSL/ASL/state.db'
video_path = '/ds/videos/opticalflow-BOBSL/ASL/videos/'
output_dir = '/ds/videos/opticalflow-BOBSL/ASL/flow/'


def mark_optical_flow_as(state, cur, con, video_id):
	res = cur.execute("UPDATE videos SET optical_flow = ?1 WHERE yt_id = ?2 AND download = 2 AND optical_flow = 1 RETURNING *", (state, video_id,))
	result = res.fetchone()
	print("mark_optical_flow_as", result)
	assert result is not None
	con.commit()


def select_and_update(new_state, cur, con, video_id):
	cur.execute("BEGIN")
	try:
		res = cur.execute("SELECT 1 FROM videos WHERE yt_id = ?1 AND download = 2 AND optical_flow = 0 LIMIT 1", (video_id,))
		result = res.fetchone()
		if result is None:
			print(f"Video {video_id} already processed!")
			cur.execute("ROLLBACK")
			return False
		res = cur.execute("UPDATE videos SET optical_flow = ?1 WHERE yt_id = ?2 AND download = 2 AND optical_flow = 0 RETURNING *", (new_state, video_id,))
		result = res.fetchone()
		print("select_and_update", result)
		assert result is not None
		cur.execute("COMMIT")
	except sqlite3.Error as e:
		print("failed!")
		cur.execute("ROLLBACK")
		raise e
	return True



def create_frame_path(video_output_dir, frame_num, flow_frame_offset):
	frame_label_int = frame_num - (flow_frame_offset//2)
	frame_label_full = f'flow_{frame_label_int:07d}.png'
	frame_path = os.path.join(video_output_dir, frame_label_full)
	return frame_label_full, frame_path


def analyze_video(parameter):
	(video, video_id) = parameter
	print('Will analyze video', video)
	file_path = os.path.join(video_path, video)
	cap = cv2.VideoCapture(file_path)
	if not cap.isOpened():
		raise Exception("Could not open video file")

	video_output_dir = os.path.join(output_dir, video_id)
	os.makedirs(video_output_dir, exist_ok=True)

	frame_list = []
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(video_id, 'FPS:', fps)

	flow_frame_offset = round(fps / 3)

	with open(os.path.join(video_output_dir, "fps.txt"), "w") as file:
		file.write(str(fps)+"\n")

	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for frame_num in range(video_length):
		if frame_num < flow_frame_offset:
			continue
		frame_label_full, frame_path = create_frame_path(video_output_dir, frame_num, flow_frame_offset)
		if not os.path.isfile(frame_path):
			print(video_id, f"Frame {frame_label_full} does not exist")
			break
	else:
		print(video_id, "Skipping video since all frames already exist")
		return video_id, True

	print(video_id, 'video_length:', video_length)

	frame_num = 0
	success = True
	while cap.isOpened():
		try:
			# Read frame from the video
			ret, prev_frame = cap.read()
			frame_list.append(prev_frame)
			if not ret:	
				print("End of video")
				success = True
				break
		except:
			continue

		frame_num += 1
		# print('frame_num:', frame_num, 'len(frame_list):', len(frame_list))
		if frame_num < flow_frame_offset:
			continue

		frame_label_full, frame_path = create_frame_path(video_output_dir, frame_num, flow_frame_offset)
		if os.path.isfile(frame_path):
			print(video_id, f"Skipping frame {frame_label_full} since it already exists")
		else:
			try:
				flow_map = flow_estimator(frame_list[0], frame_list[-1])
				flow_img = flow_estimator.draw_flow()
			except Exception as e:
				print(video_id, f"Error estimating optical flow: {e}")
				success = False
				break

			if flow_img is None or not isinstance(flow_img, np.ndarray):
				print(video_id, f"Invalid flow image at frame {frame_num}")
				success = False
				break

			print(frame_path)

			if not cv2.imwrite(frame_path, flow_img):
				print("Could not write image")
				success = False
				break
		frame_list.pop(0)

	return video_id, success


def already_processed_video_ids_first(videos):
	video_ids = []
	video_id_to_name = {}
	for video in videos:
		match = re.match(r'.*\[(.+)\]', video)
		if not match:
			raise Exception(f"Could not extract ID from filename {video}")
		(video_id,) = match.groups()
		video_id_to_name[video_id] = video
		video_ids.append(video_id)

	processed_video_ids = os.listdir(output_dir)

	sorted_video_ids = []
	for video_id in processed_video_ids:
		if video_id not in video_ids:
			raise Exception(f"Video {video_id} is in the output directory but not in the video directory")
		sorted_video_ids.append(video_id)
		video_ids.remove(video_id)

	#sorted_video_ids.extend(video_ids)

	return video_id_to_name, sorted_video_ids


con = sqlite3.connect(db_path, timeout=60)
cur = con.cursor()

class VideoIterator:
	def __iter__(self):
		videos = os.listdir(video_path)
		self.video_id_to_name, self.video_ids = already_processed_video_ids_first(videos)
		return self

	def __len__(self):
		res = cur.execute("SELECT count(*) FROM videos WHERE download = 2 AND optical_flow = 0")
		result = res.fetchone()
		assert result is not None
		(total,) = result
		return total

	def __next__(self):
		while True:
			if len(self.video_ids) == 0:
				raise StopIteration
			video_id = self.video_ids.pop(0)
			we_got_the_job = select_and_update(STATE_ESTIMATING, cur, con, video_id)
			if not we_got_the_job:
				#print("We did not get the job")
				continue
			video = self.video_id_to_name[video_id]
			return (video, video_id)


videoiteratorclass = VideoIterator()
#videoiterator = iter(videoiteratorclass)


In = TypeVar("In")
Out = TypeVar("Out")

def lazy_executor_map(
    fn: Callable[[In], Out],
    it: Iterable[In],
    ex: Executor,
    # may want this to be equal to the n_threads/n_processes 
    n_concurrent: int = 6
) -> Iterator[Out]:

    queue: list[Future[Out]] = []
    in_progress: set[Future[Out]] = set()
    itr = iter(it)

    try:
        while True:
            for _ in range(n_concurrent - len(in_progress)):
                el = next(itr) # this line will raise StopIteration when finished
                # - which will get caught by the try: except: below
                fut = ex.submit(fn, el)
                queue.append(fut)
                in_progress.add(fut)

            _, in_progress = wait(in_progress, return_when=FIRST_COMPLETED)

            # iterate over the queue, yielding outputs if available in the order they came in with
            while queue and queue[0].done():
                yield queue.pop(0).result()

    except StopIteration:
        wait(queue)
        for fut in queue:
            yield fut.result()

with ThreadPoolExecutor(max_workers=10) as ex:
	for (video_id, success) in lazy_executor_map(analyze_video, videoiteratorclass, ex, n_concurrent=10):
		print(video_id, success)
		if not success:
			print(f"Marking video {video_id} as failed")
			mark_optical_flow_as(STATE_FAILED, cur, con, video_id)
		else:
			print(f"Marking video {video_id} as completed")
			mark_optical_flow_as(STATE_COMPLETED, cur, con, video_id)
con.close()

#python -m video_flow_estimation
