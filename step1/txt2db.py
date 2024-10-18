import os
import sqlite3
import argparse

def read_video_ids(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Meaning:
# 0: not processed
# 1: processing
# 2: processed
# 3: failed

def save_to_db(yids, db_file):
    data = [(yid,) for yid in yids]

    con = sqlite3.connect(db_file, timeout=60)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            yt_id TEXT NOT NULL,
            download INTEGER NOT NULL,
            optical_flow INTEGER NOT NULL,
            features_resnet18 INTEGER NOT NULL,
            features_resnet34 INTEGER NOT NULL,
            features_resnet101 INTEGER NOT NULL,
            features_vgg16 INTEGER NOT NULL,
            UNIQUE (yt_id)
        ) STRICT
    """)
    cur.executemany("INSERT INTO videos (yt_id, download, optical_flow, features_resnet18, features_resnet34, features_resnet101, features_vgg16) VALUES (?, 0, 0, 0, 0, 0, 0)", data)
    con.commit()
    con.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids_file', type=str, help='File containing YouTube video IDs')
    parser.add_argument('--db_file', type=str, help='db file')

    args = parser.parse_args()
    yids = read_video_ids(args.ids_file)
    print(f"Found {len(yids)} videos")
    save_to_db(yids, args.db_file)


# Download https://storage.googleapis.com/gresearch/youtube-asl/youtube_asl_video_ids.txt
# python -m prep.txt2db --ids_file youtube_asl_video_ids.txt --db_file /ds/videos/opticalflow-BOBSL/ASL/state.db
