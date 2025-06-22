import os
import re
import shutil
import argparse
import tempfile
import subprocess
from tqdm import tqdm

# TODO: combine with the other file

def download_subtitle(video_id, original_subtitle_path, subtitle_dir, trash_dir, yt_dlp):
    prevdir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as dirpath:
            os.chdir(dirpath)
            cmd = [
                yt_dlp,
                "https://youtube.com/watch?v=" + video_id,
                "--skip-download",
                "--write-subs", "--write-auto-subs", "--convert-subs", "srt", "--sub-format", "srt",
                "--sub-lang", "ase",
            ]
            print(cmd)
            completed_process = subprocess.run(cmd, shell=False)
            if completed_process.returncode != 0:
                print(f"Failed to download subtitle of video {video_id}")
                return
            file_list = os.listdir()
            print(file_list)
            assert len(file_list) <= 1, f"Too many files"
            if len(file_list) <= 0:
                print(f"Failed to download subtitle {video_id}")
                return
            file = file_list[0]
            assert file.endswith(".srt")
            shutil.move(file, subtitle_dir)
            shutil.move(original_subtitle_path, trash_dir)
    finally:
        os.chdir(prevdir)


def download_subtitles(subtitle_dir, trash_dir, yt_dlp):
    os.makedirs(subtitle_dir, exist_ok=True)
    downloaded_subtitle_filenames = os.listdir(subtitle_dir)
    for downloaded_subtitle_filename in tqdm(downloaded_subtitle_filenames):
        downloaded_subtitle_path = os.path.join(subtitle_dir, downloaded_subtitle_filename)
        if os.path.getsize(downloaded_subtitle_path) > 0:
            continue
        match = re.match(r'.*\[(.+)\]', downloaded_subtitle_filename)
        if match is None:
            print(f"Failed to parse video id from {downloaded_subtitle_filename}")
            continue
        (video_id,) = match.groups()
        download_subtitle(video_id, downloaded_subtitle_path, subtitle_dir, trash_dir, yt_dlp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=str, help='subtitle directory')
    parser.add_argument('--trash', type=str, help='where to put the trash')

    args = parser.parse_args()
    print(f"Download videos into {args.dest}")
    os.makedirs(args.dest, exist_ok=True)
    os.makedirs(args.trash, exist_ok=True)
    yt_dlp = os.path.join(os.getcwd(), "yt-dlp")
    if not os.path.isfile(yt_dlp):
        raise Exception("yt-dlp binary not found in current path, please download it")
    download_subtitles(args.dest, args.trash, yt_dlp)


# python -m prep.download_replace_empty_subtitles --dest /ds/videos/opticalflow-BOBSL/ASL/subtitles --trash /ds/videos/opticalflow-BOBSL/ASL/trash
