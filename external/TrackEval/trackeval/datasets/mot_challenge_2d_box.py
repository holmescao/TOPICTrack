import os
import csv
import configparser
import pdb

import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException


class MotChallenge2DBox(_BaseDataset):
   

    @staticmethod
    def get_default_dataset_config():
      
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/mot_challenge/"
            ),  
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/mot_challenge/"
            ),  
            "OUTPUT_FOLDER": None,  
            "TRACKERS_TO_EVAL": None, 
            "CLASSES_TO_EVAL": ["pedestrian"], 
            "BENCHMARK": "MOT17",  
            "SPLIT_TO_EVAL": "train",  
            "INPUT_AS_ZIP": False,  
            "PRINT_CONFIG": True,  
            "DO_PREPROC": True,  
            "TRACKER_SUB_FOLDER": "data", 
            "OUTPUT_SUB_FOLDER": "", 
            "TRACKER_DISPLAY_NAMES": None, 
            "SEQMAP_FOLDER": None, 
            "SEQMAP_FILE": None, 
            "SEQ_INFO": None, 
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt", 
            "SKIP_SPLIT_FOL": False,  
            # "alpha_gate": 0.0, 
           
        }
        return default_config

    def __init__(self, config=None):

        super().__init__()
     
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )

        self.benchmark = self.config["BENCHMARK"]
        gt_set = self.config["BENCHMARK"] + "-" + self.config["SPLIT_TO_EVAL"]
        # gt_set1 = self.config["BENCHMARK"] + "-" + self.config["SPLIT_TO_EVAL"] + '_' + str(float((self.config["alpha_gate"])))
        self.gt_set = gt_set
        if not self.config["SKIP_SPLIT_FOL"]:
            split_fol = gt_set
        else:
            split_fol = ""
        self.gt_fol = os.path.join(self.config["GT_FOLDER"], split_fol)
        self.tracker_fol = os.path.join(self.config["TRACKERS_FOLDER"], split_fol)
        # self.tracker_fol = os.path.join(self.config["TRACKERS_FOLDER"], gt_set1)
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config["INPUT_AS_ZIP"]
        self.do_preproc = self.config["DO_PREPROC"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        self.valid_classes = ["pedestrian"]
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only pedestrian class is valid."
            )
        self.class_name_to_class_id = {
            "pedestrian": 1,
            "person_on_vehicle": 2,
            "car": 3,
            "bicycle": 4,
            "motorbike": 5,
            "non_mot_vehicle": 6,
            "static_person": 7,
            "distractor": 8,
            "occluder": 9,
            "occluder_on_ground": 10,
            "occluder_full": 11,
            "reflection": 12,
            "crowd": 13,
        }
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException("No sequences are selected to be evaluated.")

        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(
                    gt_folder=self.gt_fol, seq=seq
                )
                if not os.path.isfile(curr_file):
                    print("GT file not found " + curr_file)
                    raise TrackEvalException("GT file not found for sequence: " + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, "data.zip")
            if not os.path.isfile(curr_file):
                print("GT file not found " + curr_file)
                raise TrackEvalException(
                    "GT file not found: " + os.path.basename(curr_file)
                )

        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config["TRACKERS_TO_EVAL"] is not None) and (
            len(self.config["TRACKER_DISPLAY_NAMES"]) == len(self.tracker_list)
        ):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.config["TRACKER_DISPLAY_NAMES"])
            )
        else:
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol + ".zip"
                )
                if not os.path.isfile(curr_file):
                    print("Tracker file not found: " + curr_file)
                    raise TrackEvalException(
                        "Tracker file not found: "
                        + tracker
                        + "/"
                        + os.path.basename(curr_file)
                    )
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(
                        self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt"
                    )
                    if not os.path.isfile(curr_file):
                        print("Tracker file not found: " + curr_file)
                        raise TrackEvalException(
                            "Tracker file not found: "
                            + tracker
                            + "/"
                            + self.tracker_sub_fol
                            + "/"
                            + os.path.basename(curr_file)
                        )

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            for seq, seq_length in seq_lengths.items():
                if seq_length is None:
                    ini_file = os.path.join(self.gt_fol, seq, "seqinfo.ini")
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException(
                            "ini file does not exist: "
                            + seq
                            + "/"
                            + os.path.basename(ini_file)
                        )
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data["Sequence"]["seqLength"])

        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(
                        self.config["GT_FOLDER"], "seqmaps", self.gt_set + ".txt"
                    )
                else:
                    seqmap_file = os.path.join(
                        self.config["SEQMAP_FOLDER"], self.gt_set + ".txt"
                    )
            if not os.path.isfile(seqmap_file):
                print("no seqmap found: " + seqmap_file)
                raise TrackEvalException(
                    "no seqmap found: " + os.path.basename(seqmap_file)
                )
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == "":
                        continue
                    seq = row[0]
                    seq_list.append(seq)
                    ini_file = os.path.join(self.gt_fol, seq, "seqinfo.ini")
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException(
                            "ini file does not exist: "
                            + seq
                            + "/"
                            + os.path.basename(ini_file)
                        )
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data["Sequence"]["seqLength"])
        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
       
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, "data.zip")
            else:
                zip_file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol + ".zip"
                )
            file = seq + ".txt"
        else:
            zip_file = None
            if is_gt:
                file = self.config["GT_LOC_FORMAT"].format(
                    gt_folder=self.gt_fol, seq=seq
                )
            else:
                file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt"
                )

        read_data, ignore_data = self._load_simple_text_file(
            file, is_zipped=self.data_is_zipped, zip_file=zip_file
        )

        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_crowd_ignore_regions", "gt_extras"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = "Ground-truth"
            else:
                text = "Tracking"
            raise TrackEvalException(
                text
                + " data contains the following invalid timesteps in seq %s: " % seq
                + ", ".join([str(x) + ", " for x in extra_time_keys])
            )

        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=np.float32)
                except ValueError:
                    if is_gt:
                        raise TrackEvalException(
                            "Cannot convert gt data for sequence %s to float. Is data corrupted?"
                            % seq
                        )
                    else:
                        raise TrackEvalException(
                            "Cannot convert tracking data from tracker %s, sequence %s to float. Is data corrupted?"
                            % (tracker, seq)
                        )
                try:
                    raw_data["dets"][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data["ids"][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    if is_gt:
                        err = (
                            "Cannot load gt data from sequence %s, because there is not enough "
                            "columns in the data." % seq
                        )
                        raise TrackEvalException(err)
                    else:
                        err = (
                            "Cannot load tracker data from tracker %s, sequence %s, because there is not enough "
                            "columns in the data." % (tracker, seq)
                        )
                        raise TrackEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data["classes"][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data["classes"][t] = np.ones_like(raw_data["ids"][t])
                    else:
                        raise TrackEvalException(
                            "GT data is not in a valid format, there is not enough rows in seq %s, timestep %i."
                            % (seq, t)
                        )
                if is_gt:
                    gt_extras_dict = {
                        "zero_marked": np.atleast_1d(time_data[:, 6].astype(int))
                    }
                    raw_data["gt_extras"][t] = gt_extras_dict
                else:
                    raw_data["tracker_confidences"][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data["dets"][t] = np.empty((0, 4))
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {"zero_marked": np.empty(0)}
                    raw_data["gt_extras"][t] = gt_extras_dict
                else:
                    raw_data["tracker_confidences"][t] = np.empty(0)
            if is_gt:
                raw_data["gt_crowd_ignore_regions"][t] = np.empty((0, 4))

        if is_gt:
            key_map = {"ids": "gt_ids", "classes": "gt_classes", "dets": "gt_dets"}
        else:
            key_map = {
                "ids": "tracker_ids",
                "classes": "tracker_classes",
                "dets": "tracker_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
       
        self._check_unique_ids(raw_data)

        distractor_class_names = [
            "person_on_vehicle",
            "static_person",
            "distractor",
            "reflection",
        ]
        if self.benchmark == "MOT20":
            distractor_class_names.append("non_mot_vehicle")
        distractor_classes = [
            self.class_name_to_class_id[x] for x in distractor_class_names
        ]
        cls_id = self.class_name_to_class_id[cls]

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data["num_timesteps"]):

            gt_ids = raw_data["gt_ids"][t]
            gt_dets = raw_data["gt_dets"][t]
            gt_classes = raw_data["gt_classes"][t]
            gt_zero_marked = raw_data["gt_extras"][t]["zero_marked"]

            tracker_ids = raw_data["tracker_ids"][t]
            tracker_dets = raw_data["tracker_dets"][t]
            tracker_classes = raw_data["tracker_classes"][t]
            tracker_confidences = raw_data["tracker_confidences"][t]
            similarity_scores = raw_data["similarity_scores"][t]

            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    "Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at "
                    "timestep %i." % (np.max(tracker_classes), raw_data["seq"], t)
                )

          
            to_remove_tracker = np.array([], np.int32)
            if (
                self.do_preproc
                and self.benchmark != "MOT15"
                and gt_ids.shape[0] > 0
                and tracker_ids.shape[0] > 0
            ):

                invalid_classes = np.setdiff1d(
                    np.unique(gt_classes), self.valid_class_numbers
                )
                if len(invalid_classes) > 0:
                    print(" ".join([str(x) for x in invalid_classes]))
                    raise (
                        TrackEvalException(
                            "Attempting to evaluate using invalid gt classes. "
                            "This warning only triggers if preprocessing is performed, "
                            "e.g. not for MOT15 or where prepropressing is explicitly disabled. "
                            "Please either check your gt data, or disable preprocessing. "
                            "The following invalid classes were found in timestep "
                            + str(t)
                            + ": "
                            + " ".join([str(x) for x in invalid_classes])
                        )
                    )

                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = (
                    matching_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
                )
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(
                    gt_classes[match_rows], distractor_classes
                )
                to_remove_tracker = match_cols[is_distractor_class]

            data["tracker_ids"][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data["tracker_confidences"][t] = np.delete(
                tracker_confidences, to_remove_tracker, axis=0
            )
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            if self.do_preproc and self.benchmark != "MOT15":
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & (
                    np.equal(gt_classes, cls_id)
                )
            else:
        
                gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            data["gt_ids"][t] = gt_ids[gt_to_keep_mask]
            data["gt_dets"][t] = gt_dets[gt_to_keep_mask, :]
            data["similarity_scores"][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(np.int32)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[
                        data["tracker_ids"][t]
                    ].astype(np.int32)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(
            gt_dets_t, tracker_dets_t, box_format="xywh"
        )
        return similarity_scores
