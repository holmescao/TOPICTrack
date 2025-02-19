import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException


class KittiMOTS(_BaseDataset):
    """Dataset class for KITTI MOTS tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/kitti/kitti_mots_val"
            ),  
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/kitti/kitti_mots_val"
            ),  
           
            "OUTPUT_FOLDER": None,
          
            "TRACKERS_TO_EVAL": None,
            
            "CLASSES_TO_EVAL": ["car", "pedestrian"],
            "SPLIT_TO_EVAL": "val", 
            "INPUT_AS_ZIP": False,  
            "PRINT_CONFIG": True,  
          
            "TRACKER_SUB_FOLDER": "data",
          
            "OUTPUT_SUB_FOLDER": "",
          
            "TRACKER_DISPLAY_NAMES": None,
        
            "SEQMAP_FOLDER": None,
           
            "SEQMAP_FILE": None,
          
            "SEQ_INFO": None,
           
            "GT_LOC_FORMAT": "{gt_folder}/label_02/{seq}.txt",
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
        self.split_to_eval = self.config["SPLIT_TO_EVAL"]
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config["INPUT_AS_ZIP"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        self.valid_classes = ["car", "pedestrian"]
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. "
                "Only classes [car, pedestrian] are valid."
            )
        self.class_name_to_class_id = {
            "car": "1", "pedestrian": "2", "ignore": "10"}

        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException(
                "No sequences are selected to be evaluated.")

        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(
                    gt_folder=self.gt_fol, seq=seq
                )
                if not os.path.isfile(curr_file):
                    print("GT file not found " + curr_file)
                    raise TrackEvalException(
                        "GT file not found for sequence: " + seq)
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
        seqmap_name = "evaluate_mots.seqmap." + self.config["SPLIT_TO_EVAL"]

        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]
        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(
                        self.config["GT_FOLDER"], seqmap_name)
                else:
                    seqmap_file = os.path.join(
                        self.config["SEQMAP_FOLDER"], seqmap_name
                    )
            if not os.path.isfile(seqmap_file):
                print("no seqmap found: " + seqmap_file)
                raise TrackEvalException(
                    "no seqmap found: " + os.path.basename(seqmap_file)
                )
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, _ in enumerate(reader):
                    dialect = csv.Sniffer().sniff(fp.read(1024))
                    fp.seek(0)
                    reader = csv.reader(fp, dialect)
                    for row in reader:
                        if len(row) >= 4:
                            seq = "%04d" % int(row[0])
                            seq_list.append(seq)
                            seq_lengths[seq] = int(row[3]) + 1
        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        
        from pycocotools import mask as mask_utils

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

        if is_gt:
            crowd_ignore_filter = {2: ["10"]}
        else:
            crowd_ignore_filter = None

        read_data, ignore_data = self._load_simple_text_file(
            file,
            crowd_ignore_filter=crowd_ignore_filter,
            is_zipped=self.data_is_zipped,
            zip_file=zip_file,
            force_delimiters=" ",
        )

        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_ignore_region"]
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
     
            all_masks = []
            if time_key in read_data.keys():
                try:
                    raw_data["dets"][t] = [
                        {
                            "size": [int(region[3]), int(region[4])],
                            "counts": region[5].encode(encoding="UTF-8"),
                        }
                        for region in read_data[time_key]
                    ]
                    raw_data["ids"][t] = np.atleast_1d(
                        [region[1] for region in read_data[time_key]]
                    ).astype(int)
                    raw_data["classes"][t] = np.atleast_1d(
                        [region[2] for region in read_data[time_key]]
                    ).astype(int)
                    all_masks += raw_data["dets"][t]
                except IndexError:
                    self._raise_index_error(is_gt, tracker, seq)
                except ValueError:
                    self._raise_value_error(is_gt, tracker, seq)
            else:
                raw_data["dets"][t] = []
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
            if is_gt:
                if time_key in ignore_data.keys():
                    try:
                        time_ignore = [
                            {
                                "size": [int(region[3]), int(region[4])],
                                "counts": region[5].encode(encoding="UTF-8"),
                            }
                            for region in ignore_data[time_key]
                        ]
                        raw_data["gt_ignore_region"][t] = mask_utils.merge(
                            [mask for mask in time_ignore], intersect=False
                        )
                        all_masks += [raw_data["gt_ignore_region"][t]]
                    except IndexError:
                        self._raise_index_error(is_gt, tracker, seq)
                    except ValueError:
                        self._raise_value_error(is_gt, tracker, seq)
                else:
                    raw_data["gt_ignore_region"][t] = mask_utils.merge(
                        [], intersect=False
                    )

            if all_masks:
                masks_merged = all_masks[0]
                for mask in all_masks[1:]:
                    if (
                        mask_utils.area(
                            mask_utils.merge(
                                [masks_merged, mask], intersect=True)
                        )
                        != 0.0
                    ):
                        raise TrackEvalException(
                            "Tracker has overlapping masks. Tracker: "
                            + tracker
                            + " Seq: "
                            + seq
                            + " Timestep: "
                            + str(t)
                        )
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
        raw_data["seq"] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
       
        self._check_unique_ids(raw_data)

        cls_id = int(self.class_name_to_class_id[cls])

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data["num_timesteps"]):

            gt_class_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_dets = [
                raw_data["gt_dets"][t][ind]
                for ind in range(len(gt_class_mask))
                if gt_class_mask[ind]
            ]

            tracker_class_mask = np.atleast_1d(
                raw_data["tracker_classes"][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = [
                raw_data["tracker_dets"][t][ind]
                for ind in range(len(tracker_class_mask))
                if tracker_class_mask[ind]
            ]
            similarity_scores = raw_data["similarity_scores"][t][gt_class_mask, :][
                :, tracker_class_mask
            ]

            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 -
                                np.finfo("float").eps] = -10000
                match_rows, match_cols = linear_sum_assignment(
                    -matching_scores)
                actually_matched_mask = (
                    matching_scores[match_rows,
                                    match_cols] > 0 + np.finfo("float").eps
                )
                match_cols = match_cols[actually_matched_mask]

                unmatched_indices = np.delete(
                    unmatched_indices, match_cols, axis=0)

            unmatched_tracker_dets = [
                tracker_dets[i]
                for i in range(len(tracker_dets))
                if i in unmatched_indices
            ]
            ignore_region = raw_data["gt_ignore_region"][t]
            intersection_with_ignore_region = self._calculate_mask_ious(
                unmatched_tracker_dets, [ignore_region], is_encoded=True, do_ioa=True
            )
            is_within_ignore_region = np.any(
                intersection_with_ignore_region > 0.5 + np.finfo("float").eps, axis=1
            )

            to_remove_tracker = unmatched_indices[is_within_ignore_region]
            data["tracker_ids"][t] = np.delete(
                tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(
                tracker_dets, to_remove_tracker, axis=0)
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
        data["cls"] = cls

        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_mask_ious(
            gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False
        )
        return similarity_scores

    @staticmethod
    def _raise_index_error(is_gt, tracker, seq):
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
                "Cannot load tracker data from tracker %s, sequence %s, because there are not enough "
                "columns in the data." % (tracker, seq)
            )
            raise TrackEvalException(err)

    @staticmethod
    def _raise_value_error(is_gt, tracker, seq):
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
                "Tracking data from tracker %s, sequence %s cannot be converted to the right format. "
                "Is data corrupted?" % (tracker, seq)
            )
