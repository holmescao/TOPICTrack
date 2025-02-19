import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException
from .. import _timing
from ..datasets.rob_mots_classmap import cls_id_to_name


class RobMOTS(_BaseDataset):
    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/rob_mots"
            ), 
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/rob_mots"
            ),  
           
            "OUTPUT_FOLDER": None,
           
            "TRACKERS_TO_EVAL": None,
         
            "SUB_BENCHMARK": None,
           
            "CLASSES_TO_EVAL": None,
            "SPLIT_TO_EVAL": "train",  
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": True, 
        
            "OUTPUT_SUB_FOLDER": "results",
          
            "TRACKER_SUB_FOLDER": "data",
          
            "TRACKER_DISPLAY_NAMES": None,
         
            "SEQMAP_FOLDER": None,
           
            "SEQMAP_FILE": None,
        
            "CLSMAP_FOLDER": None,
          
            "CLSMAP_FILE": None,
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
      
        self.config = utils.init_config(
            config, self.get_default_dataset_config())

        self.split = self.config["SPLIT_TO_EVAL"]
        valid_benchmarks = [
            "mots_challenge",
            "kitti_mots",
            "bdd_mots",
            "davis_unsupervised",
            "youtube_vis",
            "ovis",
            "waymo",
            "tao",
        ]
        self.box_gt_benchmarks = ["waymo", "tao"]

        self.sub_benchmark = self.config["SUB_BENCHMARK"]
        if not self.sub_benchmark:
            raise TrackEvalException(
                "SUB_BENCHMARK config input is required (there is no default value)"
                + ", ".join(valid_benchmarks)
                + " are valid."
            )
        if self.sub_benchmark not in valid_benchmarks:
            raise TrackEvalException(
                "Attempted to evaluate an invalid benchmark: "
                + self.sub_benchmark
                + ". Only benchmarks "
                + ", ".join(valid_benchmarks)
                + " are valid."
            )

        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = os.path.join(
            self.config["TRACKERS_FOLDER"], self.config["SPLIT_TO_EVAL"]
        )
        self.data_is_zipped = self.config["INPUT_AS_ZIP"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = os.path.join(
            self.config["OUTPUT_SUB_FOLDER"], self.sub_benchmark
        )

    
        self._get_seq_info()

        if len(self.seq_list) < 1:
            raise TrackEvalException(
                "No sequences are selected to be evaluated.")

        valid_class_ids = np.atleast_1d(
            np.genfromtxt(
                os.path.join(self.gt_fol, self.split,
                             self.sub_benchmark, "clsmap.txt")
            )
        )
        valid_classes = [cls_id_to_name[int(x)]
                         for x in valid_class_ids] + ["all"]
        self.valid_class_ids = valid_class_ids
        self.class_name_to_class_id = {
            cls_name: cls_id for cls_id, cls_name in cls_id_to_name.items()
        }
        self.class_name_to_class_id["all"] = -1
        if not self.config["CLASSES_TO_EVAL"]:
            self.class_list = valid_classes
        else:
            self.class_list = [
                cls if cls in valid_classes else None
                for cls in self.config["CLASSES_TO_EVAL"]
            ]
            if not all(self.class_list):
                raise TrackEvalException(
                    "Attempted to evaluate an invalid class. Only classes "
                    + ", ".join(valid_classes)
                    + " are valid."
                )

        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = os.path.join(
                    self.gt_fol, self.split, self.sub_benchmark, "data", seq + ".txt"
                )
                if not os.path.isfile(curr_file):
                    print("GT file not found " + curr_file)
                    raise TrackEvalException(
                        "GT file not found for sequence: " + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(
                self.gt_fol, self.split, self.sub_benchmark, "data.zip"
            )
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
                curr_file = os.path.join(self.tracker_fol, tracker, "data.zip")
                if not os.path.isfile(curr_file):
                    raise TrackEvalException(
                        "Tracker file not found: " +
                        os.path.basename(curr_file)
                    )
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(
                        self.tracker_fol,
                        tracker,
                        self.tracker_sub_fol,
                        self.sub_benchmark,
                        seq + ".txt",
                    )
                    if not os.path.isfile(curr_file):
                        print("Tracker file not found: " + curr_file)
                        raise TrackEvalException(
                            "Tracker file not found: "
                            + self.sub_benchmark
                            + "/"
                            + os.path.basename(curr_file)
                        )

    def get_name(self):
        return self.get_class_name() + "." + self.sub_benchmark

    def _get_seq_info(self):
        self.seq_list = []
        self.seq_lengths = {}
        self.seq_sizes = {}
        self.seq_ignore_class_ids = {}
        if self.config["SEQMAP_FILE"]:
            seqmap_file = self.config["SEQMAP_FILE"]
        else:
            if self.config["SEQMAP_FOLDER"] is None:
                seqmap_file = os.path.join(
                    self.gt_fol, self.split, self.sub_benchmark, "seqmap.txt"
                )
            else:
                seqmap_file = os.path.join(
                    self.config["SEQMAP_FOLDER"], self.split + ".seqmap"
                )
        if not os.path.isfile(seqmap_file):
            print("no seqmap found: " + seqmap_file)
            raise TrackEvalException(
                "no seqmap found: " + os.path.basename(seqmap_file)
            )
        with open(seqmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.readline(), delimiters=" ")
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for i, row in enumerate(reader):
                if len(row) >= 4:
                  
                    seq = row[0]
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[1])
                    self.seq_sizes[seq] = (int(row[2]), int(row[3]))
                    self.seq_ignore_class_ids[seq] = [
                        int(row[x]) for x in range(4, len(row))
                    ]

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
       
        from pycocotools import mask as mask_utils

        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(
                    self.gt_fol, self.split, self.sub_benchmark, "data.zip"
                )
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, "data.zip")
            file = seq + ".txt"
        else:
            zip_file = None
            if is_gt:
                file = os.path.join(
                    self.gt_fol, self.split, self.sub_benchmark, "data", seq + ".txt"
                )
            else:
                file = os.path.join(
                    self.tracker_fol,
                    tracker,
                    self.tracker_sub_fol,
                    self.sub_benchmark,
                    seq + ".txt",
                )

        read_data, ignore_data = self._load_simple_text_file(
            file, is_zipped=self.data_is_zipped, zip_file=zip_file, force_delimiters=" "
        )

        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if not is_gt:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for t in range(num_timesteps):
            time_key = str(t)
          
            all_valid_masks = []
            if time_key in read_data.keys():
                try:
                    raw_data["ids"][t] = np.atleast_1d(
                        [det[1] for det in read_data[time_key]]
                    ).astype(int)
                    raw_data["classes"][t] = np.atleast_1d(
                        [det[2] for det in read_data[time_key]]
                    ).astype(int)
                    if (not is_gt) or (
                        self.sub_benchmark not in self.box_gt_benchmarks
                    ):
                        raw_data["dets"][t] = [
                            {
                                "size": [int(region[4]), int(region[5])],
                                "counts": region[6].encode(encoding="UTF-8"),
                            }
                            for region in read_data[time_key]
                        ]
                        all_valid_masks += [
                            mask
                            for mask, cls in zip(
                                raw_data["dets"][t], raw_data["classes"][t]
                            )
                            if cls < 100
                        ]
                    else:
                        raw_data["dets"][t] = np.atleast_2d(
                            [det[4:8] for det in read_data[time_key]]
                        ).astype(float)

                    if not is_gt:
                        raw_data["tracker_confidences"][t] = np.atleast_1d(
                            [det[3] for det in read_data[time_key]]
                        ).astype(float)
                except IndexError:
                    self._raise_index_error(is_gt, self.sub_benchmark, seq)
                except ValueError:
                    self._raise_value_error(is_gt, self.sub_benchmark, seq)
          
            else:
                if (not is_gt) or (self.sub_benchmark not in self.box_gt_benchmarks):
                    raw_data["dets"][t] = []
                else:
                    raw_data["dets"][t] = np.empty((0, 4)).astype(float)
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if not is_gt:
                    raw_data["tracker_confidences"][t] = np.empty(
                        0).astype(float)

            if all_valid_masks:
                masks_merged = all_valid_masks[0]
                for mask in all_valid_masks[1:]:
                    if (
                        mask_utils.area(
                            mask_utils.merge(
                                [masks_merged, mask], intersect=True)
                        )
                        != 0.0
                    ):
                        err = "Overlapping masks in frame %d" % t
                        raise TrackEvalException(err)
                    masks_merged = mask_utils.merge(
                        [masks_merged, mask], intersect=False
                    )

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
        raw_data["frame_size"] = self.seq_sizes[seq]
        raw_data["seq"] = seq
        return raw_data

    @staticmethod
    def _raise_index_error(is_gt, sub_benchmark, seq):
        """
        Auxiliary method to raise an evaluation error in case of an index error while reading files.
        :param is_gt: whether gt or tracker data is read
        :param tracker: the name of the tracker
        :param seq: the name of the seq
        :return: None
        """
        if is_gt:
            err = (
                "Cannot load gt data from sequence %s, because there are not enough "
                "columns in the data." % seq
            )
            raise TrackEvalException(err)
        else:
            err = (
                "Cannot load tracker data from benchmark %s, sequence %s, because there are not enough "
                "columns in the data." % (sub_benchmark, seq)
            )
            raise TrackEvalException(err)

    @staticmethod
    def _raise_value_error(is_gt, sub_benchmark, seq):
        """
        Auxiliary method to raise an evaluation error in case of an value error while reading files.
        :param is_gt: whether gt or tracker data is read
        :param tracker: the name of the tracker
        :param seq: the name of the seq
        :return: None
        """
        if is_gt:
            raise TrackEvalException(
                "GT data for sequence %s cannot be converted to the right format. Is data corrupted?"
                % seq
            )
        else:
            raise TrackEvalException(
                "Tracking data from benchmark %s, sequence %s cannot be converted to the right format. "
                "Is data corrupted?" % (sub_benchmark, seq)
            )

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
       
        from pycocotools import mask as mask_utils

        self._check_unique_ids(raw_data)

        cls_id = self.class_name_to_class_id[cls]
        ignore_class_id = cls_id + 100
        seq = raw_data["seq"]

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

            if cls == "all":
                gt_class_mask = raw_data["gt_classes"][t] < 100
          
            elif self.sub_benchmark == "waymo" and cls == "car":
                waymo_vehicle_classes = np.array([3, 4, 6, 8])
                gt_class_mask = np.isin(
                    raw_data["gt_classes"][t], waymo_vehicle_classes
                )
            else:
                gt_class_mask = raw_data["gt_classes"][t] == cls_id
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            if cls == "all":
                ignore_regions_mask = raw_data["gt_classes"][t] >= 100
            else:
                ignore_regions_mask = raw_data["gt_classes"][t] == ignore_class_id
                ignore_regions_mask = np.logical_or(
                    ignore_regions_mask, raw_data["gt_classes"][t] == 100
                )
            if self.sub_benchmark in self.box_gt_benchmarks:
                gt_dets = raw_data["gt_dets"][t][gt_class_mask]
                ignore_regions_box = raw_data["gt_dets"][t][ignore_regions_mask]
                if len(ignore_regions_box) > 0:
                    ignore_regions_box[:, 2] = (
                        ignore_regions_box[:, 2] - ignore_regions_box[:, 0]
                    )
                    ignore_regions_box[:, 3] = (
                        ignore_regions_box[:, 3] - ignore_regions_box[:, 1]
                    )
                    ignore_regions = mask_utils.frPyObjects(
                        ignore_regions_box,
                        self.seq_sizes[seq][0],
                        self.seq_sizes[seq][1],
                    )
                else:
                    ignore_regions = []
            else:
                gt_dets = [
                    raw_data["gt_dets"][t][ind]
                    for ind in range(len(gt_class_mask))
                    if gt_class_mask[ind]
                ]
                ignore_regions = [
                    raw_data["gt_dets"][t][ind]
                    for ind in range(len(ignore_regions_mask))
                    if ignore_regions_mask[ind]
                ]

            if cls == "all":
                tracker_class_mask = np.ones_like(
                    raw_data["tracker_classes"][t])
            else:
                tracker_class_mask = np.atleast_1d(
                    raw_data["tracker_classes"][t] == cls_id
                )
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = [
                raw_data["tracker_dets"][t][ind]
                for ind in range(len(tracker_class_mask))
                if tracker_class_mask[ind]
            ]
            tracker_confidences = raw_data["tracker_confidences"][t][tracker_class_mask]
            similarity_scores = raw_data["similarity_scores"][t][gt_class_mask, :][
                :, tracker_class_mask
            ]
            tracker_classes = raw_data["tracker_classes"][t][tracker_class_mask]

            if tracker_ids.shape[0] > 0:

                unmatched_indices = np.arange(tracker_ids.shape[0])
                if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                    matching_scores = similarity_scores.copy()
                    matching_scores[matching_scores <
                                    0.5 - np.finfo("float").eps] = 0
                    match_rows, match_cols = linear_sum_assignment(
                        -matching_scores)
                    actually_matched_mask = (
                        matching_scores[match_rows, match_cols]
                        > 0 + np.finfo("float").eps
                    )
               
                    match_cols = match_cols[actually_matched_mask]
                    unmatched_indices = np.delete(
                        unmatched_indices, match_cols, axis=0)

              

                if cls_id in self.seq_ignore_class_ids[seq]:
                  
                    to_remove_tracker = unmatched_indices
                else:
                    unmatched_tracker_dets = [
                        tracker_dets[i]
                        for i in range(len(tracker_dets))
                        if i in unmatched_indices
                    ]

                    tracker_boxes_t = mask_utils.toBbox(unmatched_tracker_dets)
                    unmatched_widths = tracker_boxes_t[:, 2]
                    unmatched_heights = tracker_boxes_t[:, 3]
                    unmatched_size = np.maximum(
                        unmatched_heights, unmatched_widths)
                    min_size = np.min(self.seq_sizes[seq]) / 8
                    is_too_small = unmatched_size <= min_size + \
                        np.finfo("float").eps

                    if ignore_regions:
                        ignore_region_merged = ignore_regions[0]
                        for mask in ignore_regions[1:]:
                            ignore_region_merged = mask_utils.merge(
                                [ignore_region_merged, mask], intersect=False
                            )
                        intersection_with_ignore_region = self._calculate_mask_ious(
                            unmatched_tracker_dets,
                            [ignore_region_merged],
                            is_encoded=True,
                            do_ioa=True,
                        )
                        is_within_ignore_region = np.any(
                            intersection_with_ignore_region
                            > 0.5 + np.finfo("float").eps,
                            axis=1,
                        )
                        to_remove_tracker = unmatched_indices[
                            np.logical_or(
                                is_too_small, is_within_ignore_region)
                        ]
                    else:
                        to_remove_tracker = unmatched_indices[is_too_small]

                if cls == "all":
                    unmatched_tracker_classes = [
                        tracker_classes[i]
                        for i in range(len(tracker_classes))
                        if i in unmatched_indices
                    ]
                    is_ignore_class = np.isin(
                        unmatched_tracker_classes, self.seq_ignore_class_ids[seq]
                    )
                    is_not_evaled_class = np.logical_not(
                        np.isin(unmatched_tracker_classes,
                                self.valid_class_ids)
                    )
                    to_remove_all = unmatched_indices[
                        np.logical_or(is_ignore_class, is_not_evaled_class)
                    ]
                    to_remove_tracker = np.concatenate(
                        [to_remove_tracker, to_remove_all], axis=0
                    )
            else:
                to_remove_tracker = np.array([], dtype=np.int_)

            data["tracker_ids"][t] = np.delete(
                tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(
                tracker_dets, to_remove_tracker, axis=0)
            data["tracker_confidences"][t] = np.delete(
                tracker_confidences, to_remove_tracker, axis=0
            )
            similarity_scores = np.delete(
                similarity_scores, to_remove_tracker, axis=1)

            data["gt_ids"][t] = gt_ids
            data["gt_dets"][t] = gt_dets
            data["similarity_scores"][t] = similarity_scores

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
        data["frame_size"] = raw_data["frame_size"]

        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):

        from pycocotools import mask as mask_utils

        if self.sub_benchmark in self.box_gt_benchmarks:

            tracker_boxes_t = mask_utils.toBbox(tracker_dets_t)
            tracker_boxes_t[:, 2] = tracker_boxes_t[:, 0] + \
                tracker_boxes_t[:, 2]
            tracker_boxes_t[:, 3] = tracker_boxes_t[:, 1] + \
                tracker_boxes_t[:, 3]
            similarity_scores = self._calculate_box_ious(
                gt_dets_t, tracker_boxes_t, box_format="x0y0x1y1"
            )
        else:
            similarity_scores = self._calculate_mask_ious(
                gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False
            )
        return similarity_scores
