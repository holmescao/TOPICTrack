"""
STP: Simplest Tracker Possible

Author: Jonathon Luiten

This simple tracker, simply assigns track IDs which maximise the 'bounding box IoU' between previous tracks and current
detections. It is also able to match detections to tracks at more than one timestep previously.
"""

from trackeval.utils import get_code_path
from trackeval.baselines import baseline_utils as butils
import os
import sys
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

code_path = get_code_path()
config = {
    "INPUT_FOL": os.path.join(
        code_path, "data/detections/rob_mots/{split}/non_overlap_supplied/data/"
    ),
    "OUTPUT_FOL": os.path.join(code_path, "data/trackers/rob_mots/{split}/STP/data/"),
    "SPLIT": "train",  
    "Benchmarks": None, 
    "Num_Parallel_Cores": None,  
    "DETECTION_THRESHOLD": 0.5,
    "ASSOCIATION_THRESHOLD": 1e-10,
    "MAX_FRAMES_SKIP": 7,
}


def track_sequence(seq_file):

    data = butils.load_seq(seq_file)

    output_data = []

    curr_max_id = 0

    for cls, cls_data in data.items():

        prev = {
            "boxes": np.empty((0, 4)),
            "ids": np.array([], np.int_),
            "timesteps": np.array([]),
        }

        for timestep, t_data in enumerate(cls_data):

            t_data = butils.threshold(t_data, config["DETECTION_THRESHOLD"])

            boxes = butils.masks2boxes(
                t_data["mask_rles"], t_data["im_hs"], t_data["im_ws"]
            )

            ious = butils.box_iou(prev["boxes"], boxes)

            prev_timestep_scores = np.power(10, -1 * prev["timesteps"])

            match_scores = prev_timestep_scores[:, np.newaxis] * ious

            match_rows, match_cols = butils.match(match_scores)

            actually_matched_mask = (
                ious[match_rows, match_cols] > config["ASSOCIATION_THRESHOLD"]
            )
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            ids = np.nan * np.ones((len(boxes),), np.int_)
            ids[match_cols] = prev["ids"][match_rows]

            num_not_matched = len(ids) - len(match_cols)
            new_ids = np.arange(
                curr_max_id + 1, curr_max_id + num_not_matched + 1)
            ids[np.isnan(ids)] = new_ids

            curr_max_id += num_not_matched

       
            unmatched_rows = [
                i
                for i in range(len(prev["ids"]))
                if i not in match_rows
                and (prev["timesteps"][i] + 1 <= config["MAX_FRAMES_SKIP"])
            ]

            prev["ids"] = np.concatenate(
                (ids, prev["ids"][unmatched_rows]), axis=0)
            prev["boxes"] = np.concatenate(
                (np.atleast_2d(boxes), np.atleast_2d(
                    prev["boxes"][unmatched_rows])),
                axis=0,
            )
            prev["timesteps"] = np.concatenate(
                (np.zeros((len(ids),)), prev["timesteps"][unmatched_rows] + 1), axis=0
            )

            for i in range(len(t_data["ids"])):
                row = [
                    timestep,
                    int(ids[i]),
                    cls,
                    t_data["scores"][i],
                    t_data["im_hs"][i],
                    t_data["im_ws"][i],
                    t_data["mask_rles"][i],
                ]
                output_data.append(row)

    out_file = seq_file.replace(
        config["INPUT_FOL"].format(split=config["SPLIT"]),
        config["OUTPUT_FOL"].format(split=config["SPLIT"]),
    )
    butils.write_seq(output_data, out_file)

    print("DONE:", seq_file)


if __name__ == "__main__":

    freeze_support()

    if config["Benchmarks"]:
        benchmarks = config["Benchmarks"]
    else:
        benchmarks = [
            "davis_unsupervised",
            "kitti_mots",
            "youtube_vis",
            "ovis",
            "bdd_mots",
            "tao",
        ]
        if config["SPLIT"] != "train":
            benchmarks += ["waymo", "mots_challenge"]
    seqs_todo = []
    for bench in benchmarks:
        bench_fol = os.path.join(
            config["INPUT_FOL"].format(split=config["SPLIT"]), bench
        )
        seqs_todo += [os.path.join(bench_fol, seq)
                      for seq in os.listdir(bench_fol)]

    if config["Num_Parallel_Cores"]:
        with Pool(config["Num_Parallel_Cores"]) as pool:
            results = pool.map(track_sequence, seqs_todo)

    else:
        for seq_todo in seqs_todo:
            track_sequence(seq_todo)
