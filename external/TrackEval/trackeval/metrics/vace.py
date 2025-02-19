import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class VACE(_BaseMetric):
  

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ["VACE_IDs", "VACE_GT_IDs", "num_non_empty_timesteps"]
        self.float_fields = ["STDA", "ATA", "FDA", "SFDA"]
        self.fields = self.integer_fields + self.float_fields
        self.summary_fields = ["SFDA", "ATA"]

        self._additive_fields = self.integer_fields + ["STDA", "FDA"]

        self.threshold = 0.5

    @_timing.time
    def eval_sequence(self, data):
       
        res = {}

        potential_matches_count = np.zeros(
            (data["num_gt_ids"], data["num_tracker_ids"])
        )
        gt_id_count = np.zeros(data["num_gt_ids"])
        tracker_id_count = np.zeros(data["num_tracker_ids"])
        both_present_count = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"])
        ):
            
            matches_mask = np.greater_equal(
                data["similarity_scores"][t], self.threshold
            )
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[
                gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]
            ] += 1
          
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1
            both_present_count[
                gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]
            ] += 1
     
        union_count = (
            gt_id_count[:, np.newaxis]
            + tracker_id_count[np.newaxis, :]
            - both_present_count
        )
     
        with np.errstate(divide="raise", invalid="raise"):
            temporal_iou = potential_matches_count / union_count
       
        match_rows, match_cols = linear_sum_assignment(-temporal_iou)
        res["STDA"] = temporal_iou[match_rows, match_cols].sum()
        res["VACE_IDs"] = data["num_tracker_ids"]
        res["VACE_GT_IDs"] = data["num_gt_ids"]

        non_empty_count = 0
        fda = 0
        for t, (gt_ids_t, tracker_ids_t) in enumerate(
            zip(data["gt_ids"], data["tracker_ids"])
        ):
            n_g = len(gt_ids_t)
            n_d = len(tracker_ids_t)
            if not (n_g or n_d):
                continue
          
            non_empty_count += 1
            if not (n_g and n_d):
                continue
           
            spatial_overlap = data["similarity_scores"][t]
            match_rows, match_cols = linear_sum_assignment(-spatial_overlap)
            overlap_ratio = spatial_overlap[match_rows, match_cols].sum()
            fda += overlap_ratio / (0.5 * (n_g + n_d))
        res["FDA"] = fda
        res["num_non_empty_timesteps"] = non_empty_count

        res.update(self._compute_final_fields(res))
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=True):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.fields:
            if ignore_empty_classes:
                res[field] = np.mean(
                    [
                        v[field]
                        for v in all_res.values()
                        if v["VACE_GT_IDs"] > 0 or v["VACE_IDs"] > 0
                    ],
                    axis=0,
                )
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self._additive_fields:
            res[field] = _BaseMetric._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
      
        res = {}
        for header in self._additive_fields:
            res[header] = _BaseMetric._combine_sum(all_res, header)
        res.update(self._compute_final_fields(res))
        return res

    @staticmethod
    def _compute_final_fields(additive):
        final = {}
        with np.errstate(invalid="ignore"): 
            final["ATA"] = additive["STDA"] / (
                0.5 * (additive["VACE_IDs"] + additive["VACE_GT_IDs"])
            )
            final["SFDA"] = additive["FDA"] / additive["num_non_empty_timesteps"]
        return final
