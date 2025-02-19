
from __future__ import absolute_import
import numpy as np

from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter
import time
import copy
import logging
import numba as nb
from . import tracker


INFTY_COST = 1e18



def get_remain_pair_matrix(cost_matrix, row_indices):
   

    track_ids = np.arange(cost_matrix.shape[0])
    row_indices = np.array(row_indices)
    if row_indices.shape[0] > 0:
        remain_track_ids = np.delete(track_ids, row_indices)
        cost_matrix = cost_matrix[remain_track_ids, :]

    return cost_matrix


def better_np_unique(arr):
    

    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, _, counts = np.unique(arr,
                                               return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes, counts


def split_conflict(pre_matches):
   
    vals, indexes, counts = better_np_unique(pre_matches[:, 0])

    single_idxs = np.where(counts == 1)[0]
    single_det_indices = [int(indexes[i]) for i in single_idxs]
    matches = pre_matches[single_det_indices]

    conflict_matches = []
    multi_idxs = np.where(counts > 1)[0]
    for i in multi_idxs:
        conflict_matches.append(((pre_matches[indexes[i], 0]), vals[i]))

    return list(map(tuple, matches)), conflict_matches


def split_TD_conflict(pre_matches):
   
    T_vals, T_indexes, T_counts = better_np_unique(pre_matches[:, 0])
    D_vals, D_indexes, D_counts = better_np_unique(pre_matches[:, 1])

    matches, matches_ind = [], []
    
    single_idxs = np.where(T_counts == 1)[0]
    for i in single_idxs:
        T_ind = int(T_indexes[i])
        dv = pre_matches[T_ind, 1]
        c = D_counts[np.argwhere(D_vals == dv)]
        if c == 1:
            matches.append(pre_matches[T_ind, :2])
            matches_ind.append(T_ind)

    two_idxs = np.where(T_counts == 2)[0]
    for i in two_idxs:
        T_ind0 = int(T_indexes[i][0])
        T_ind1 = int(T_indexes[i][1])
        dv0 = pre_matches[T_ind0, 1]
        dv1 = pre_matches[T_ind1, 1]
        if dv0 == dv1:
            matches.append(pre_matches[T_ind0, :2])
            matches_ind.append(T_ind0)
            matches_ind.append(T_ind1)
    matches = np.array(matches, dtype=np.int)
   
    conflicts_ind = list(
        set(np.arange(pre_matches.shape[0])) - set(matches_ind))
    conflict_matches = pre_matches[conflicts_ind, :]

    return list(map(tuple, matches)), conflict_matches


def _assign(level, cost_matrix, track_info, track_indices, detection_indices, gate):
    cost_matrix = filter_pairs(cost_matrix, gate)
    row_indices, col_indices = linear_assignment(cost_matrix)

    pre_matches = []
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] <= gate:
            pre_matches.append(
                (track_idx, detection_idx, track_info[track_idx]))
    pre_matches = np.array(pre_matches)

    if pre_matches.shape[0]:
        unmatched_tracks = list(set(track_indices) -
                                set(list(pre_matches[:, 0])))
        unmatched_detections = list(set(detection_indices) -
                                    set(list(pre_matches[:, 1])))
        if level == 0:
            pre_matches = np.hstack(
                (pre_matches, np.zeros(pre_matches.shape[0]).reshape(-1, 1)))
        else:
            pre_matches = np.hstack(
                (pre_matches, np.ones(pre_matches.shape[0]).reshape(-1, 1)))
    else:
        unmatched_tracks = track_indices
        unmatched_detections = detection_indices

    return pre_matches, unmatched_tracks, unmatched_detections



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


def cost_metric(metric, sigma, level, tracks, dets, track_indices, detection_indices):
    features = np.array([dets[i].feature for i in detection_indices])
    targets = np.array([tracks[i].track_id for i in track_indices])
    tracks = [tracks[i] for i in track_indices]
    dets_use = [dets[i] for i in detection_indices]

    det_previous_ct = np.array([d.previous_ct for d in dets_use])
    track_previous_ct = np.array([t.ct for t in tracks])

    delta_xy = np.array([t.mean[4:6] for t in tracks])
    alpha = tracker.metric_gaussian_motion(delta_xy, sigma)
    if level == 0:
        start_time = time.time()
        
        motion_cost = dist_cost_matrix(
            track_previous_ct, det_previous_ct, tracks, dets_use)
        motion_time = time.time()
        motion_t = motion_time - start_time
       
        cost_matrix = motion_cost
    else:
        
        start_time = time.time()
        appearance_cost = metric.distance(features, targets)
        appear_time = time.time()
        appear_t = appear_time - start_time
       
        cost_matrix = appearance_cost

    tracks_info = {}
    for i, idx in enumerate(track_indices):
        tracks_info[idx] = alpha[i]

    return cost_matrix, tracks_info


def metric_assign(metric, sigma, level, tracks, detections, track_indices, detection_indices, metric_gate):
  
    start_time = time.time()
   
    cost_matrix, info = cost_metric(metric,
                                    sigma, level, tracks, detections, track_indices, detection_indices)

    dis_time = time.time()
    dis_t = dis_time - start_time

    matches, unmatched_tracks, unmatched_detections = _assign(
        level, cost_matrix, info, track_indices, detection_indices, metric_gate)
    assign_time = time.time()
    assign_t = assign_time - dis_time
  
    return (matches, unmatched_tracks, unmatched_detections)



def test(a):
    time.sleep(0.002)
    a = np.array([(1, 1, 1, 1)] * 15)
    return a, a, a


def pre_assignment(opt, level, tracks, detections, track_indices, detection_indices, metric, motion_distance, appearance_distance):
   

    if level == 0:
        
        start_time = time.time()
        results = []
        for i, gate in enumerate([motion_distance, appearance_distance]):
            results.append(
                metric_assign(metric, opt.sigma, i, tracks, detections, track_indices, detection_indices, gate))

        results = [r for r in results]

        single_time = time.time()
        single_t = single_time - start_time
       

        motion_results = results[0]
        appearance_results = results[1]

        pre_matches = np.vstack((motion_results[0], appearance_results[0])
                                ) if appearance_results[0].shape[0] else motion_results[0]
        unmatched_tracks = list(set(motion_results[1]).intersection(
            set(appearance_results[1])))
        unmatched_detections = list(set(motion_results[2]).intersection(
            set(appearance_results[2])))

        return pre_matches, unmatched_tracks, unmatched_detections
    else:
        
        appearance_matches, appearance_unmatched_tracks, appearance_unmatched_detections = metric_assign(
            metric, opt.sigma, 1, tracks, detections, track_indices, detection_indices, appearance_distance)

        return appearance_matches, list(appearance_unmatched_tracks), list(appearance_unmatched_detections)


def pre_assignmentv1(cost_matrix, tracks, detections, track_indices, detection_indices, gate, max_distance, distance_metric):
  
    unmatched_track_indices = copy.deepcopy(track_indices)
    pre_matches, unmatched_tracks, unmatched_detections, = [], [], []
   
    while len(unmatched_track_indices):
        row_indices, col_indices = linear_assignment(cost_matrix)

        cur_assignment_idx = []
        for row, col in zip(row_indices, col_indices):
            track_idx = unmatched_track_indices[row]
            detection_idx = detection_indices[col]

            if cost_matrix[row, col] > gate:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                pre_matches.append((track_idx, detection_idx))

            cur_assignment_idx.append(track_idx)

        for idx in cur_assignment_idx:
            unmatched_track_indices.remove(idx)

        cost_matrix = get_remain_pair_matrix(cost_matrix, row_indices)

        if len(unmatched_track_indices):
            gate = max_distance

            cost_matrix, _ = distance_metric(
                1, tracks, detections, unmatched_tracks, unmatched_detections)
            cost_matrix = filter_pairs(cost_matrix, gate=gate)

    pre_matches = np.array(sorted(pre_matches, key=lambda p: p[0]))

    return pre_matches, unmatched_tracks


def extract_conflict_tracks_info(tracklet_idx, det_id_index, tracks_info, alpha_thredhold):
    fisrt_tracks, second_tracks = [], []
    for idx in tracklet_idx:
        appearance_cost = tracks_info[idx]["appearance_cost"][det_id_index]
       
        miss_nums = tracks_info[idx]["miss_nums"]
        if tracks_info[idx]['alpha'] >= alpha_thredhold:
            fisrt_tracks.append((idx, appearance_cost,  miss_nums))
          
        else:
            second_tracks.append(
                (idx, appearance_cost, miss_nums))
        
    fisrt_tracks = np.array(fisrt_tracks)
    second_tracks = np.array(second_tracks)

    return fisrt_tracks, second_tracks


def first_round_association(fisrt_tracks, second_tracks, det_id, tolerate_threshold):
   
    if len(fisrt_tracks) == 0:
        return [], []

    if second_tracks.shape[0] > 0:
        min_second_appearance = np.min(second_tracks[:, 1])
        accepted_appearance = np.where((fisrt_tracks[:, 1] - min_second_appearance) /
                                       min_second_appearance <= tolerate_threshold, True, False)
    else:
        accepted_appearance = np.array(fisrt_tracks.shape[0] * [True])

    matches, unmatched_tracks = [], []
  
    matches_tracks_ids = fisrt_tracks[accepted_appearance, 0]
    matches = [(int(i), det_id) for i in matches_tracks_ids]
 
    unmatched_tracks = list(fisrt_tracks[~accepted_appearance, 0])

    if len(matches):
      
        return [matches[0]], unmatched_tracks
    else:
        return matches, unmatched_tracks


def second_round_association(second_tracks, det_id, max_discount_num, det_used):
   
    matches, unmatched_tracks, kf_ids = [], [], []
    while second_tracks.shape[0]:
        if not det_used:
       
            min_id = np.argmin(second_tracks[:, 1])
            track_id = second_tracks[min_id, 0]
            matches.append((int(track_id), det_id))
            
            remain_ids = [True] * second_tracks.shape[0]
            remain_ids[min_id] = False
            second_tracks = second_tracks[remain_ids, :]
        else:
            discount = np.where(
                second_tracks[:, -1] <= max_discount_num, True, False)
          
            kf_ids = list(second_tracks[discount, 0].astype(np.int32))
            unmatched_tracks = list(second_tracks[~discount, 0])

            break

    return matches, unmatched_tracks, kf_ids


def two_round_match_v1(conflict_matches, tracks_info, detection_indices, alpha_thredhold=0.5, tolerate_threshold=0.1, max_discount_num=10):

    matches, unmatched_tracks, kf_ids = [], [], []
    for pair in conflict_matches:
        tracklet_idx, det_id = pair[0], pair[1]

        det_id_index = detection_indices.index(det_id)
        fisrt_tracks, second_tracks = extract_conflict_tracks_info(
            tracklet_idx, det_id_index, tracks_info, alpha_thredhold)

        matches_first, unmatched_tracks_first = first_round_association(
            fisrt_tracks, second_tracks, det_id, tolerate_threshold)

        det_used = False if len(matches_first) == 0 else True
        matches_second, unmatched_tracks_second, kf_ids = second_round_association(
            second_tracks, det_id, max_discount_num, det_used)

        matches = matches_first + matches_second
        unmatched_tracks = unmatched_tracks_first + unmatched_tracks_second

    return matches, unmatched_tracks, kf_ids


def filter_pairs(cost_matrix, gate):
    cost_matrix[cost_matrix > gate] = gate + INFTY_COST

    return cost_matrix


def iou_cost_matching(
    distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
 
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices 

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)

    cost_matrix[cost_matrix > max_distance] = max_distance + INFTY_COST

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def select_pairs(matches, col_id, pairs):
    for t in pairs:
        ind = np.argwhere(matches[:, col_id] == t)
        if len(ind):
            matches[ind[0], -1] = 0
    matches = matches[matches[:, -1] == 1, :]

    return matches


def two_round_match(conflicts, alpha_gate=0.9):
    if conflicts.shape[0] == 0:
        return [], [], []
    
    first_round = conflicts[conflicts[:, -2] >= alpha_gate, :]
    matches_a = first_round[first_round[:, -1] == 1, :2]

    second_round = conflicts[conflicts[:, -2] < alpha_gate, :]
    second_round = second_round[second_round[:, -1] == 0, :]
    mask = np.ones(second_round.shape[0]).reshape(-1, 1)
    second_round = np.hstack((second_round, mask))

    second_round = select_pairs(second_round, 0, matches_a[:, 0])

    second_round = select_pairs(second_round, 1, matches_a[:, 1])
    matches_b = second_round[:, :2]

    matches_ = np.vstack((matches_a, matches_b))
    matches_ = matches_.astype(np.int)
    unmatched_tracks_ = list(set(conflicts[:, 0]) - set(matches_[:, 0]))
    unmatched_tracks_ = [int(i) for i in unmatched_tracks_]
    unmatched_detections_ = list(set(conflicts[:, 1]) - set(matches_[:, 1]))
    unmatched_detections_ = [int(i) for i in unmatched_detections_]

    return list(map(tuple, matches_)), unmatched_tracks_, unmatched_detections_


def min_cost_matching(
        opt, level, metric, appearance_distance, tracks, detections, track_indices=None,
        detection_indices=None, motion_distance=1e+18):
   
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices, []  

   
    start_time = time.time()
    pre_matches, unmatched_tracks, unmatched_detections = pre_assignment(opt, level, tracks, detections,
                                                                         track_indices, detection_indices,
                                                                         metric, motion_distance, appearance_distance)
    preassign_time = time.time()
    preassign_t = preassign_time - start_time

    if pre_matches.shape[0]:
        matches, conflicts = split_TD_conflict(pre_matches)
        conflict_time = time.time()
        conflict_t = conflict_time - preassign_time

        matches_, unmatched_tracks_, unmatched_detections_ = two_round_match(
            conflicts, opt.alpha_gate)
        tworound_time = time.time()
        tworound_t = tworound_time - conflict_time
        tot_t = time.time() - start_time
        matches += matches_
        unmatched_tracks += unmatched_tracks_
        unmatched_detections += unmatched_detections_
        kf_ids = []
      
    else:
        matches = []
        kf_ids = []

    matches = sorted(matches, key=lambda m: m[0])
    return matches, unmatched_tracks, unmatched_detections, kf_ids
   


def matching_cascade(opt,
                     metric, max_distance, cascade_depth, tracks, detections,
                     track_indices=None, detection_indices=None):
  
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches, kf_ids = [], []
   
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0: 
            break
       
        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0: 
            continue

        matches_l, _, unmatched_detections, kf_ids_l = min_cost_matching(opt,
                                                                         level, metric, max_distance, tracks, detections,
                                                                         track_indices_l, unmatched_detections)

        matches += matches_l
        kf_ids += kf_ids_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, unmatched_detections, kf_ids


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
   
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def motion_cost_matrix(
        kf, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
   
    def normalization(data):
    
        if data.shape[0] == 1:
            data[0, 0] = 1.0
            return data
        else:
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        track = tracks[row]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)

        cost_matrix[row, :] = gating_distance
       

    return normalization(cost_matrix)
