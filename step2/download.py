import os
import re
import shutil
import sqlite3
import argparse
import tempfile
import subprocess
from tqdm import tqdm

STATE_DOWNLOADING = 1
STATE_COMPLETED = 2
STATE_FAILED = 3


def mark_download_as(state, cur, con, video_id):
    res = cur.execute("UPDATE videos SET download = ?1 WHERE yt_id = ?2 AND download = 1 RETURNING *", (state, video_id,))
    result = res.fetchone()
    print(result)
    assert result is not None
    con.commit()


def select_and_update(new_state, cur, con):
    video_id = None
    cur.execute("BEGIN")
    try:
        res = cur.execute("SELECT yt_id FROM videos WHERE download = 0 LIMIT 1")
        result = res.fetchone()
        if result is None:
            print("Done!")
            cur.execute("ROLLBACK")
            return None
        (video_id,) = result
        res = cur.execute("UPDATE videos SET download = ?1 WHERE yt_id = ?2 AND download = 0 RETURNING *", (new_state, video_id,))
        result = res.fetchone()
        print(result)
        assert result is not None
        cur.execute("COMMIT")
    except sql.Error as e:
        print("failed!")
        cur.execute("ROLLBACK")
        con.close()
        raise e
    return video_id


def update_progress(cur, con, t):
    assert update_progress.previous_total is not None
    res = cur.execute("SELECT count(*) FROM videos WHERE download = 0")
    result = res.fetchone()
    assert result is not None
    (current_total,) = result
    t.update(update_progress.previous_total - current_total)
    update_progress.previous_total = current_total
update_progress.previous_total = None


def download_videos(db_file, target_dir, yt_dlp):
    con = sqlite3.connect(db_file, timeout=60)
    cur = con.cursor()

    video_dir = os.path.join(target_dir, "videos")
    subtitle_dir = os.path.join(target_dir, "subtitles")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(subtitle_dir, exist_ok=True)
    downloaded_video_paths = os.listdir(video_dir)
    for downloaded_video_path in downloaded_video_paths:
        match = re.match(r'.*\[(.+)\]', downloaded_video_path)
        if match is None:
            print(f"Failed to parse video id from {downloaded_video_path}")
            continue
        (downloaded_video_id,) = match.groups()
        cur.execute("UPDATE videos SET download = 2 WHERE yt_id = ?1 AND download = 0", (downloaded_video_id,))
        con.commit()
    prevdir = os.getcwd()
    try:
        res = cur.execute("SELECT count(*) FROM videos WHERE download = 0")
        result = res.fetchone()
        assert result is not None
        (initial_total,) = result
        update_progress.previous_total = initial_total
        t = tqdm(total=initial_total)
        while True:
            video_id = select_and_update(STATE_DOWNLOADING, cur, con)
            if video_id is None:
                break

            with tempfile.TemporaryDirectory() as dirpath:
                os.chdir(dirpath)
                cmd = [
                    yt_dlp,
                    "https://youtube.com/watch?v=" + video_id,
                    "--write-subs", "--write-auto-subs", "--convert-subs", "srt", "--sub-format", "srt",
                ]
                print(cmd)
                completed_process = subprocess.run(cmd, shell=False)
                if completed_process.returncode != 0:
                    print(f"Failed to download video {video_id}")
                    mark_download_as(STATE_FAILED, cur, con, video_id)
                    update_progress(cur, con, t)
                    continue
                file_list = os.listdir()
                print(file_list)
                assert len(file_list) <= 2, f"Too many files"
                saved_subtitle = False
                saved_video = False
                for file in file_list:
                    if file.endswith(".srt"):
                        assert not saved_subtitle
                        shutil.move(file, subtitle_dir)
                        saved_subtitle = True
                    else:
                        assert not saved_video
                        shutil.move(file, video_dir)
                        saved_video = True
                if not saved_video:
                    print(f"Failed to download video {video_id}")
                    mark_download_as(STATE_FAILED, cur, con, video_id)
                    update_progress(cur, con, t)
                    continue
                if not saved_subtitle:
                    print(f"Failed to download subtitle {video_id}")
                    mark_download_as(STATE_FAILED, cur, con, video_id)
                    update_progress(cur, con, t)
                    continue

            mark_download_as(STATE_COMPLETED, cur, con, video_id)
            update_progress(cur, con, t)
        t.close()
    finally:
        os.chdir(prevdir)
    con.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_file', type=str, help='Database containing YouTube video IDs')
    parser.add_argument('--dest', type=str, help='Destination directory')

    args = parser.parse_args()
    print(f"Download videos into {args.dest}")
    os.makedirs(args.dest, exist_ok=True)
    yt_dlp = os.path.join(os.getcwd(), "yt-dlp")
    if not os.path.isfile(yt_dlp):
        raise Exception("yt-dlp binary not found in current directory, please download it")
    download_videos(args.db_file, args.dest, yt_dlp)


# python -m prep.download --db_file /ds/videos/opticalflow-BOBSL/ASL/state.db --dest /ds/videos/opticalflow-BOBSL/ASL
