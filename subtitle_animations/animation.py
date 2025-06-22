import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from datetime import datetime, timedelta
from tqdm import tqdm
matplotlib.use('TkAgg')


# Function to parse SRT file
def parse_srt(srt_file):
    subtitles = []
    with open(srt_file, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = re.compile(r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.+?)(?=\n\n|\Z)",
                         re.DOTALL)
    matches = re.findall(pattern, content)

    # Convert timings and store sentences
    for match in matches:
        start_time = datetime.strptime(match[1], '%H:%M:%S,%f')
        end_time = datetime.strptime(match[2], '%H:%M:%S,%f')
        sentence = match[3].replace("\n", " ").strip()
        subtitles.append({'start': start_time, 'end': end_time, 'sentence': sentence})

    return subtitles


# Filter out empty sentences
def filter_subtitles(subtitles):
    return [s for s in subtitles if s['sentence']]


# Plot subtitle segments with animation for the time cursor
def plot_subtitles_with_cursor(subtitles1, subtitles2, save_path=None):
    dpi = 100
    width_px = 1280
    height_px = width_px / 3
    w_in_inches = width_px / dpi
    h_in_inches = height_px / dpi

    fig, ax = plt.subplots(figsize=(w_in_inches, h_in_inches))  # Adjusted height to display two rows

    # Create the plot for the first subtitle stream (blue, y=0)
    for i, subtitle in enumerate(subtitles1):
        start_time = mdates.date2num(subtitle['start'])
        end_time = mdates.date2num(subtitle['end'])
        ax.barh(0, end_time - start_time, left=start_time, height=0.4, color='blue',
                label='GT' if i == 0 else "")
        # Add a vertical line at the start of each subtitle
        ax.axvline(start_time, color='blue', linestyle='--', alpha=0.6)

    # Create the plot for the second subtitle stream (green, y=1)
    for i, subtitle in enumerate(subtitles2):
        start_time = mdates.date2num(subtitle['start'])
        end_time = mdates.date2num(subtitle['end'])
        ax.barh(1, end_time - start_time, left=start_time, height=0.4, color='green',
                label='Predicted' if i == 0 else "")
        # Add a vertical line at the start of each subtitle
        ax.axvline(start_time, color='green', linestyle='--', alpha=0.6)

    # Formatting the plot
    ax.set_yticks([0, 1])  # Two rows: y=0 for Set 1, y=1 for Set 2
    ax.set_yticklabels(['GT', 'Predicted'])  # Labels for each row
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    start_video_time = datetime(1900, 1, 1, 0, 0, 0)
    ax.set_xlim(left=start_video_time)
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.title("Video Subtitle Segments")

    # Create a vertical line (cursor) that will move across the plot
    cursor, = ax.plot([], [], 'r-', lw=2, label='Time Cursor')

    total_duration = max(
        (subtitles1[-1]['end'] - start_video_time).total_seconds(),
        (subtitles2[-1]['end'] - start_video_time).total_seconds()
    )

    fps = 29.9700300  # Output video's frame rate
    frames = round(total_duration * fps)
    print(f"Total duration: {total_duration} seconds")
    print(f"Total frames: {frames}")

    pbar = tqdm(total=frames, unit='f')

    # Function to update the cursor position
    def update(frame):
        # Smooth cursor movement: divide the movement into smaller steps
        current_time_seconds = total_duration * (frame / (frames - 1))  # Position as fraction of total duration

        current_time = update.start_video_time + timedelta(seconds=current_time_seconds) # red line started from start of the video i.e 00:00:00
        cursor.set_data([mdates.date2num(current_time), mdates.date2num(current_time)],
                        [-0.5, 1.5])  # Cursor spans both rows
        update.pbar.update(frame - update.prev_frame)
        update.prev_frame = frame
        return cursor,

    update.start_video_time = start_video_time
    update.prev_frame = 0
    update.pbar = pbar


    plt.tight_layout()
    plt.legend()

    # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=total_duration, blit=True)

    if save_path:
        if save_path.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'Matplotlib'}, bitrate=1800)
            ani.save(save_path, writer=writer, dpi=dpi)
        elif save_path.endswith('.gif'):
            try:
                writer = PillowWriter(fps=fps)
                ani.save(save_path, writer=writer, dpi=dpi)
                print(f"Animation successfully saved to {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")

    pbar.close()

    plt.show()


# Main function to load two SRT files and plot segments
def main(srt_file1, srt_file2, save_path=None):
    subtitles1 = parse_srt(srt_file1)
    subtitles2 = parse_srt(srt_file2)

    filtered_subtitles1 = filter_subtitles(subtitles1)
    filtered_subtitles2 = filter_subtitles(subtitles2)

    plot_subtitles_with_cursor(filtered_subtitles1, filtered_subtitles2, save_path)


# Example usage
if __name__ == "__main__":
    srt_file1 = "asl/short_video/original.srt" 
    srt_file2 = "asl/short_video/generated.srt" 
    save_path = "asl/short_video/subtitle_short_asl.mp4"
    main(srt_file1, srt_file2, save_path)
