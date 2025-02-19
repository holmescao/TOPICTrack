import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException
from .. import _timing


class Kitti2DBox(_BaseDataset):
    """Dataset class for KITTI 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/kitti/kitti_2d_box_train"
            ),  
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/kitti/kitti_2d_box_train/"
            ), 
     
            "OUTPUT_FOLDER": None,
  
            "TRACKERS_TO_EVAL": None,
          
            "CLASSES_TO_EVAL": ["car", "pedestrian"],
            "SPLIT_TO_EVAL": "training",  
            "INPUT_AS_ZIP": False, 
            "PRINT_CONFIG": True, 
         
            "TRACKER_SUB_FOLDER": "data",
          
            "OUTPUT_SUB_FOLDER": "",
       
            "TRACKER_DISPLAY_NAMES": None,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
      
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )
        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config["INPUT_AS_ZIP"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        self.max_occlusion = 2
        self.max_truncation = 0
        self.min_height = 25

        self.valid_classes = ["car", "pedestrian"]
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only classes [car, pedestrian] are valid."
            )
        self.class_name_to_class_id = {
            "car": 1,
            "van": 2,
            "truck": 3,
            "pedestrian": 4,
            "person": 5,  
            "cyclist": 6,
            "tram": 7,
            "misc": 8,
            "dontcare": 9,
            "car_2": 1,
        }

        self.seq_list = []
        self.seq_lengths = {}
        seqmap_name = "evaluate_tracking.seqmap." + \
            self.config["SPLIT_TO_EVAL"]
        seqmap_file = os.path.join(self.gt_fol, seqmap_name)
        if not os.path.isfile(seqmap_file):
            raise TrackEvalException(
                "no seqmap found: " + os.path.basename(seqmap_file)
            )
        with open(seqmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.read(1024))
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for row in reader:
                if len(row) >= 4:
                    seq = row[0]
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[3])
                    if not self.data_is_zipped:
                        curr_file = os.path.join(
                            self.gt_fol, "label_02", seq + ".txt")
                        if not os.path.isfile(curr_file):
                            raise TrackEvalException(
                                "GT file not found: " +
                                os.path.basename(curr_file)
                            )
            if self.data_is_zipped:
                curr_file = os.path.join(self.gt_fol, "data.zip")
                if not os.path.isfile(curr_file):
                    raise TrackEvalException(
                        "GT file not found: " + os.path.basename(curr_file)
                    )

        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.tracker_list))
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
                file = os.path.join(self.gt_fol, "label_02", seq + ".txt")
            else:
                file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt"
                )

        if is_gt:
            crowd_ignore_filter = {2: ["dontcare"]}
        else:
            crowd_ignore_filter = None

        valid_filter = {2: [x for x in self.class_list]}
        if is_gt:
            if "car" in self.class_list:
                valid_filter[2].append("van")
            if "pedestrian" in self.class_list:
                valid_filter[2] += ["person"]

        convert_filter = {2: self.class_name_to_class_id}

        read_data, ignore_data = self._load_simple_text_file(
            file,
            time_col=0,
            id_col=1,
            remove_negative_ids=True,
            valid_filter=valid_filter,
            crowd_ignore_filter=crowd_ignore_filter,
            convert_filter=convert_filter,
            is_zipped=self.data_is_zipped,
            zip_file=zip_file,
        )
    
        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_crowd_ignore_regions", "gt_extras"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        current_time_keys = [str(t) for t in range(num_timesteps)]
        extra_time_keys = [
            x for x in read_data.keys() if x not in current_time_keys]
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
            time_key = str(t)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=np.float)
                raw_data["dets"][t] = np.atleast_2d(time_data[:, 6:10])
                raw_data["ids"][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data["classes"][t] = np.atleast_1d(
                    time_data[:, 2]).astype(int)
                if is_gt:
                    gt_extras_dict = {
                        "truncation": np.atleast_1d(time_data[:, 3].astype(int)),
                        "occlusion": np.atleast_1d(time_data[:, 4].astype(int)),
                    }
                    raw_data["gt_extras"][t] = gt_extras_dict
                else:
                    if time_data.shape[1] > 17:
                        raw_data["tracker_confidences"][t] = np.atleast_1d(
                            time_data[:, 17]
                        )
                    else:
                        raw_data["tracker_confidences"][t] = np.ones(
                            time_data.shape[0])
            else:
                raw_data["dets"][t] = np.empty((0, 4))
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {
                        "truncation": np.empty(0),
                        "occlusion": np.empty(0),
                    }
                    raw_data["gt_extras"][t] = gt_extras_dict
                else:
                    raw_data["tracker_confidences"][t] = np.empty(0)
            if is_gt:
                if time_key in ignore_data.keys():
                    time_ignore = np.asarray(
                        ignore_data[time_key], dtype=np.float)
                    raw_data["gt_crowd_ignore_regions"][t] = np.atleast_2d(
                        time_ignore[:, 6:10]
                    )
                else:
                    raw_data["gt_crowd_ignore_regions"][t] = np.empty((0, 4))

        if is_gt:
            key_map = {"ids": "gt_ids",
                       "classes": "gt_classes", "dets": "gt_dets"}
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
        """Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        KITTI:
            In KITTI, the 4 preproc steps are as follow:
                1) There are two classes (pedestrian and car) which are evaluated separately.
                2) For the pedestrian class, the 'person' class is distractor objects (people sitting).
                    For the car class, the 'van' class are distractor objects.
                    GT boxes marked as having occlusion level > 2 or truncation level > 0 are also treated as
                        distractors.
                3) Crowd ignore regions are used to remove unmatched detections. Also unmatched detections with
                    height <= 25 pixels are removed.
                4) Distractor gt dets (including truncated and occluded) are removed.
        """
        if cls == "pedestrian":
            distractor_classes = [self.class_name_to_class_id["person"]]
        elif cls == "car":
            distractor_classes = [self.class_name_to_class_id["van"]]
        else:
            raise (TrackEvalException("Class %s is not evaluatable" % cls))
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

            gt_class_mask = np.sum(
                [raw_data["gt_classes"][t] == c for c in [
                    cls_id] + distractor_classes],
                axis=0,
            )
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_dets = raw_data["gt_dets"][t][gt_class_mask]
            gt_classes = raw_data["gt_classes"][t][gt_class_mask]
            gt_occlusion = raw_data["gt_extras"][t]["occlusion"][gt_class_mask]
            gt_truncation = raw_data["gt_extras"][t]["truncation"][gt_class_mask]

            tracker_class_mask = np.atleast_1d(
                raw_data["tracker_classes"][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = raw_data["tracker_dets"][t][tracker_class_mask]
            tracker_confidences = raw_data["tracker_confidences"][t][tracker_class_mask]
            similarity_scores = raw_data["similarity_scores"][t][gt_class_mask, :][
                :, tracker_class_mask
            ]

            to_remove_matched = np.array([], np.int_)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores <
                                0.5 - np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(
                    -matching_scores)
                actually_matched_mask = (
                    matching_scores[match_rows,
                                    match_cols] > 0 + np.finfo("float").eps
                )
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(
                    gt_classes[match_rows], distractor_classes
                )
                is_occluded_or_truncated = np.logical_or(
                    gt_occlusion[match_rows]
                    > self.max_occlusion + np.finfo("float").eps,
                    gt_truncation[match_rows]
                    > self.max_truncation + np.finfo("float").eps,
                )
                to_remove_matched = np.logical_or(
                    is_distractor_class, is_occluded_or_truncated
                )
                to_remove_matched = match_cols[to_remove_matched]
                unmatched_indices = np.delete(
                    unmatched_indices, match_cols, axis=0)

            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            unmatched_heights = (
                unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
            )
            is_too_small = unmatched_heights <= self.min_height + \
                np.finfo("float").eps

            crowd_ignore_regions = raw_data["gt_crowd_ignore_regions"][t]
            intersection_with_ignore_region = self._calculate_box_ious(
                unmatched_tracker_dets,
                crowd_ignore_regions,
                box_format="x0y0x1y1",
                do_ioa=True,
            )
            is_within_crowd_ignore_region = np.any(
                intersection_with_ignore_region > 0.5 + np.finfo("float").eps, axis=1
            )

            to_remove_unmatched = unmatched_indices[
                np.logical_or(is_too_small, is_within_crowd_ignore_region)
            ]
            to_remove_tracker = np.concatenate(
                (to_remove_matched, to_remove_unmatched), axis=0
            )
            data["tracker_ids"][t] = np.delete(
                tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(
                tracker_dets, to_remove_tracker, axis=0)
            data["tracker_confidences"][t] = np.delete(
                tracker_confidences, to_remove_tracker, axis=0
            )
            similarity_scores = np.delete(
                similarity_scores, to_remove_tracker, axis=1)

            gt_to_keep_mask = (
                (np.less_equal(gt_occlusion, self.max_occlusion))
                & (np.less_equal(gt_truncation, self.max_truncation))
                & (np.equal(gt_classes, cls_id))
            )
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
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"]
                                                  [t]].astype(np.int_)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(
                len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[
                        data["tracker_ids"][t]
                    ].astype(np.int_)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(
            gt_dets_t, tracker_dets_t, box_format="x0y0x1y1"
        )
        return similarity_scores
