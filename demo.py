import os
import argparse
import cv2
import time
from mvextractor.videocap import VideoCap
import numpy as np
import atexit
import json
import subprocess
import ast
import pandas as pd


class MP4Writer:
    def __init__(self, out_filename, width, height, fps):
        atexit.register(self.release)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
        self._width = width
        self._height = height

    def release(self):
        self._writer.release()

    def write(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, _ = img.shape
        assert self._height == h and self._width == w

        self._writer.write(img)


def get_video_info(file_path):
    command = 'ffprobe -print_format json -show_streams -v quiet {}'.format(file_path)
    retcode, stdout, _ = do_cmd(command)
    print(retcode, stdout)
    if retcode == 0:
        try:
            obj = json.loads(stdout.decode("utf-8"))
        except TypeError as e:
            print(command)
            print(stdout)
            sys.exit(e)
        if len(obj["streams"]) > 0:
            return obj["streams"][0]

    return {}


def do_cmd(cmd):
    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = child.communicate()  # stdout stderr include '\n'
    ret = child.returncode
    return ret, res[0], res[1]


def get_avg_motion(motion_vectors):
    # Sum of euclidean distance of motion vectors
    return np.mean(np.sqrt(np.power(motion_vectors[::, 6] - motion_vectors[::, 4], 2) + np.power(motion_vectors[::, 5] - motion_vectors[::, 3], 2)))


def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
    
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)

    return frame


def get_video_motion(video_path, output_folder, visualize=False):
    """
    Computes the average frame motion using the video motion vector
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError("Unable to find file")

    cap = VideoCap()
    ret = cap.open(video_path)
    data = {"frame_no": [], "motion": []}

    if not ret:
        raise RuntimeError(f"Could not open {video_path}")

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    if visualize:
        vid_info = get_video_info(video_path)
        out_video_name = os.path.join(output_folder, f"{video_basename}_mv.mp4")
        writer = MP4Writer(
            out_filename=out_video_name,
            width=vid_info["width"],
            height=vid_info["height"],
            fps=round(ast.literal_eval(vid_info["avg_frame_rate"]), 0),
        )

    frame_counter = 0

    while True:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if not ret:
            break

        frame_motion = get_avg_motion(motion_vectors)
        data["frame_no"].append(frame_counter)
        data["motion"].append(frame_motion)
        frame_counter += 1
        if visualize:
            cv2.putText(frame, str(round(frame_motion, 3)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            frame = draw_motion_vectors(frame, motion_vectors)
            writer.write(frame)

    cap.release()
    if visualize:
        writer.release()

    out_df = pd.DataFrame(data)

    output_path = os.path.join(output_folder, f"{video_basename}.csv")
    out_df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="Input file", required=True)
    parser.add_argument("-o", "--output_folder", help="Output folder", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    vid_path = os.path.abspath(args.input_file)
    print(vid_path)
    print(os.getcwd())

    get_video_motion(vid_path, args.output_folder, visualize=False)