
from __future__ import absolute_import
import datetime
from sklearn.metrics.pairwise import cosine_similarity as cosine
import numpy as np
import time
import numba as nb
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilterv1()
        self.tracks = []
        self.next_id = 0
        self.cur_date = datetime.date.today()

        self.feat_update = 0.1
        self.nID = 10000
        self.embedding_bank = np.zeros((self.nID, 128)) 
        self.tracklet_ages = np.zeros((self.nID), dtype=np.int)

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)


    def update(self, detections, opt):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        start_time = time.time()
     
        matches, unmatched_tracks, unmatched_detections, kf_ids = \
            self._match(detections, opt)
        match_time = time.time()
        match_t = match_time - start_time

        
        for track_idx, detection_idx in matches:
            
            track = detections[detection_idx]
            self.embedding_bank[track_idx, :] = self.feat_update * track.feature \
                + (1-self.feat_update) * self.embedding_bank[track_idx, :]
           
            self.tracks[track_idx].update(self.kf, track)
        for track_idx in unmatched_tracks:
         
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            track = detections[detection_idx]
            if track.confidence > 0.3:  
                self._initiate_track(track)
        for track_idx in kf_ids:
            self.tracks[track_idx].update(self.kf, None)
         

        update_time = time.time()
        update_t = update_time - match_time

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]

        self.tracklet_ages[:self.next_id] = self.tracklet_ages[:self.next_id] + 1
        for track_id in active_targets:
            self.tracklet_ages[track_id - 1] = 1

        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        update_dis_time = time.time()
        update_dis_t = update_dis_time - update_time
        tot_time = time.time()
        tot_t = tot_time - start_time
      

    def _match(self, detections, opt):

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections, kf_ids_a = \
            linear_assignment.matching_cascade(opt,
                                               self.metric, self.metric.matching_threshold, self.max_age,
                                               self.tracks, detections, confirmed_tracks)
    
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.iou_cost_matching(
            iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        kf_ids = kf_ids_a
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections, kf_ids

    def _initiate_track(self, detection):
       
        embedding = detection.feature
        max_id, max_cos = self.get_similarity(embedding)

    
        window_size = 2  
        if max_cos >= 0.3 and self.tracklet_ages[max_id - 1] < window_size:
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks.append(Track(detection.ct, detection.tlwh, mean, covariance,
                                     max_id, self.n_init, self.max_age, embedding))
            self.embedding_bank[max_id-1, :] = self.feat_update * embedding \
                + (1-self.feat_update) * self.embedding_bank[max_id-1, :]
        else:
            self.next_id += 1
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks.append(Track(detection.ct, detection.tlwh, mean, covariance,
                                     self.next_id, self.n_init, self.max_age, embedding))
            self.embedding_bank[self.next_id-1, :] = embedding
      

    def get_similarity(self, feat):
       
        max_id = -1
        max_cos = -1
        nID = self.next_id

        alive = [t.track_id for t in self.tracks if t.is_confirmed()]

        a = feat[None, :]  
        b = self.embedding_bank[:nID, :]  
        if len(b) > 0:
            alive = np.array(alive, dtype=np.int) - 1 
            cosim = cosine(a, b)
            cosim = np.reshape(cosim, newshape=(-1))
            cosim[alive] = -2
            cosim[nID-1] = -2
           
            max_id = int(np.argmax(cosim) + 1)
            max_cos = np.max(cosim)

        return max_id, max_cos

    def cost_metric(self, sigma, level, tracks, dets, track_indices, detection_indices):
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        tracks = [tracks[i] for i in track_indices]
        dets_use = [dets[i] for i in detection_indices]

        det_previous_ct = np.array([d.previous_ct for d in dets_use])
        track_previous_ct = np.array([t.ct for t in tracks])

        alpha = metric_gaussian_motion(tracks, sigma)
        if level == 0:
            
            motion_cost = dist_cost_matrix(
                track_previous_ct, det_previous_ct, tracks, dets_use)

            cost_matrix = motion_cost
          
        else:
            
            appearance_cost = self.metric.distance(features, targets)

            cost_matrix = appearance_cost

        tracks_info = {}
        for i, idx in enumerate(track_indices):
            tracks_info[idx] = alpha[i]
           

        return cost_matrix, tracks_info


def dist_cost_matrix(track_previous_ct, det_previous_ct, tracks, dets, motion_gate=1.1e18, overlap_thresh=0.05):
    cost_matrix = (((det_previous_ct.reshape(1, -1, 2) -
                   track_previous_ct.reshape(-1, 1, 2)) ** 2).sum(axis=2))  
   
    tracks_wh = np.array([track.tlwh[2:4] for track in tracks])
    track_size = tracks_wh[:, 0] * tracks_wh[:, 1]
    det_wh = np.array([det.tlwh[2:4] for det in dets])
    item_size = det_wh[:, 0] * det_wh[:, 1]

    track_boxes = np.array([[track.tlwh[0],
                           track.tlwh[1],
                           track.tlwh[0]+track.tlwh[2],
                           track.tlwh[1]+track.tlwh[3]]
                            for track in tracks], np.float32) 
    det_boxes = np.array([[det.tlwh[0],
                         det.tlwh[1],
                         det.tlwh[0]+det.tlwh[2],
                         det.tlwh[1]+det.tlwh[3]]
                          for det in dets], np.float32)
    box_ious = bbox_overlaps_py(track_boxes, det_boxes)

    invalid = ((cost_matrix > track_size.reshape(-1, 1))
               + (cost_matrix > item_size.reshape(1, -1))
               + (box_ious < overlap_thresh)
               ) > 0
    cost_matrix = cost_matrix + invalid * motion_gate

    return cost_matrix


def bbox_overlaps_py(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (
            query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * \
                        (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


@nb.jit()
def metric_gaussian_motion(delta_xy, sigma=19.77, ratio_thredhold=100):
   

    alpha = 1. - np.exp(- (delta_xy[:, 0] ** 2 +
                           delta_xy[:, 1]**2) / (2 * sigma**2))

    return alpha.astype(np.float32)