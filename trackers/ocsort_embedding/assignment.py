from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn.functional as F
import sys
INFTY_COST = 999


def metric_gaussian_motion(alpha, sigma):

    return alpha.astype(np.float32)


def iou_batch(bboxes1, bboxes2):

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) *
        (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) *
        (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o


def softmax_by_row(arr):
    exp_arr = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    softmax_result = exp_arr / np.sum(exp_arr, axis=1, keepdims=True)
    return softmax_result


def calculate_aspect_ratios(det, trk):
    def aspect_ratio(bbox):
        width = bbox[:, 2] - bbox[:, 0]
        height = bbox[:, 3] - bbox[:, 1]
        return width / height

    det_aspect_ratios = aspect_ratio(det)
    trk_aspect_ratios = aspect_ratio(trk)

    abs_diff_aspect_ratios = np.abs(
        trk_aspect_ratios[:, np.newaxis] - det_aspect_ratios)

    return abs_diff_aspect_ratios


def set_non_min_to_one_by_row(arr, val=1):

    min_values = np.min(arr, axis=1, keepdims=True)

    mask = arr > min_values

    arr[mask] = val

    return arr


def appearance_associate(dets_embs,
                         trk_embs,
                         dets, tracks,
                         track_indices, detection_indices, tracks_info,
                         gate, iou_threshold, metric, rotate=False):

    if (len(dets_embs) == 0) or len(trk_embs) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(dets_embs)))

    cost_matrix, sim_matrix = cal_cost_matrix(dets_embs, trk_embs, metric)

    if rotate:
        abs_diff_aspect_ratios = calculate_aspect_ratios(dets, tracks)

        abs_diff_aspect_ratios = set_non_min_to_one_by_row(
            abs_diff_aspect_ratios)
        abs_diff_aspect_ratios[abs_diff_aspect_ratios > 0.03] = 1
        abs_diff_aspect_ratios[abs_diff_aspect_ratios <= 0.03] = gate-0.0001

    if rotate:
        cost_matrix[cost_matrix ==
                    INFTY_COST] = abs_diff_aspect_ratios[cost_matrix == INFTY_COST]

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    pre_matches = []
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]

        if cost_matrix[row, col] <= gate:
            pre_matches.append(
                (track_idx, detection_idx, tracks_info[track_idx]))

    pre_matches = np.array(pre_matches)

    if pre_matches.shape[0]:
        unmatched_tracks = list(set(track_indices) -
                                set(list(pre_matches[:, 0])))
        unmatched_detections = list(set(detection_indices) -
                                    set(list(pre_matches[:, 1])))

        pre_matches = np.hstack(
            (pre_matches, np.ones(pre_matches.shape[0]).reshape(-1, 1)))
    else:
        unmatched_tracks = track_indices
        unmatched_detections = detection_indices

    return (pre_matches, unmatched_tracks, unmatched_detections),  abs(sim_matrix.T)


def min_cost_matching(
        motion_pre_assign, appearance_pre_assign,
        alpha_gate=0.9, two_round_off=None):

    pre_matches, unmatched_tracks, unmatched_detections = \
        pre_match_process(motion_pre_assign, appearance_pre_assign)
    if pre_matches.shape[0]:

        matches, conflicts = split_TD_conflict(pre_matches)

        matches_, unmatched_tracks_, unmatched_detections_ = two_round_match(
            conflicts, alpha_gate)

        matches += matches_
        unmatched_tracks += unmatched_tracks_
        unmatched_detections += unmatched_detections_
    else:
        matches = np.empty((0, 2), dtype=int)

    matches = sorted(matches, key=lambda m: m[0])

    return np.array(matches), np.array(unmatched_tracks), np.array(unmatched_detections)


def cal_cost_matrix(dets_embs, trk_embs, metric):

    if metric:
        cost_matrix, sim_matrix = _nn_res_recons_cosine_distance(
            trk_embs, dets_embs, data_is_normalized=False)
    else:
        cost_matrix, sim_matrix = _cosine_distance(trk_embs, dets_embs)

    return cost_matrix, sim_matrix


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    sim = np.dot(a, b.T)
    return 1.-sim, sim


def reconsdot_distance(x, y, tmp=100):
    x = torch.from_numpy(np.asarray(x)).half().cuda()
    y = torch.from_numpy(np.asarray(y)).cuda()

    track_features = F.normalize(x, dim=1)
    det_features = F.normalize(y, dim=1)

    ndet, ndim, nsd = det_features.shape
    ntrk, _, nst = track_features.shape

    fdet = det_features.permute(0, 2, 1).reshape(-1, ndim).cuda()
    ftrk = track_features.permute(0, 2, 1).reshape(-1, ndim).cuda()

    aff = torch.mm(ftrk, fdet.transpose(0, 1))
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0, 1)

    recons_ftrk = torch.einsum('tds,dsm->tdm', aff_td.view(ntrk*nst, ndet, nsd),
                               fdet.view(ndet, nsd, ndim))
    recons_fdet = torch.einsum('dts,tsm->dtm', aff_dt.view(ndet*nsd, ntrk, nst),
                               ftrk.view(ntrk, nst, ndim))

    recons_ftrk = recons_ftrk.permute(0, 2, 1).reshape(ntrk, nst*ndim, ndet)

    recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    recons_fdet = recons_fdet.permute(0, 2, 1).reshape(ndet, nsd*ndim, ntrk)

    recons_fdet_norm = F.normalize(recons_fdet, dim=1)

    dot_td = torch.einsum('tad,ta->td', recons_ftrk_norm,
                          F.normalize(ftrk.reshape(ntrk, nst*ndim), dim=1))
    dot_dt = torch.einsum('dat,da->dt', recons_fdet_norm,
                          F.normalize(fdet.reshape(ndet, nsd*ndim), dim=1))

    cost_matrix = 1 - 0.5 * (dot_td + dot_dt.transpose(0, 1))
    cost_matrix = cost_matrix.detach().cpu().numpy()

    return cost_matrix


def _nn_res_recons_cosine_distance(x, y, tmp=100, data_is_normalized=False):
    if not data_is_normalized:
        x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
        y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)

    ftrk = torch.from_numpy(np.asarray(x)).half().cuda()
    fdet = torch.from_numpy(np.asarray(y)).half().cuda()
    aff = torch.mm(ftrk, fdet.transpose(0, 1))
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0, 1)

    res_recons_ftrk = torch.mm(aff_td, fdet)
    res_recons_fdet = torch.mm(aff_dt, ftrk)

    sim = (torch.mm(ftrk, fdet.transpose(0, 1)) + torch.mm(res_recons_ftrk,
                                                           res_recons_fdet.transpose(0, 1))) / 2
    distances = 1-sim

    distances = distances.detach().cpu().numpy()
    sim = sim.detach().cpu().numpy()

    return distances, sim


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

            matches.append(pre_matches[T_ind, :])

            matches_ind.append(T_ind)

    two_idxs = np.where(T_counts == 2)[0]
    for i in two_idxs:
        T_ind0 = int(T_indexes[i][0])
        T_ind1 = int(T_indexes[i][1])
        dv0 = pre_matches[T_ind0, 1]
        dv1 = pre_matches[T_ind1, 1]
        if dv0 == dv1:

            matches.append(pre_matches[T_ind0, :])

            matches_ind.append(T_ind0)
            matches_ind.append(T_ind1)

    conflicts_ind = list(
        set(np.arange(pre_matches.shape[0])) - set(matches_ind))
    conflict_matches = pre_matches[conflicts_ind, :]

    return list(map(tuple, matches)), conflict_matches


def pre_match_process(motion_results, appearance_results):

    if motion_results[0].shape[0]:
        if appearance_results[0].shape[0]:
            pre_matches = np.vstack((np.insert(motion_results[0], 4, 0, axis=1), np.insert(
                appearance_results[0], 4, 1, axis=1)))
        else:
            pre_matches = np.insert(motion_results[0], 4, 0, axis=1)

        unmatched_tracks = list(set(motion_results[1]).intersection(
            set(appearance_results[1])))
        unmatched_detections = list(set(motion_results[2]).intersection(
            set(appearance_results[2])))
    else:

        pre_matches = appearance_results[0]

        if len(pre_matches) == 0:
            pass
        else:
            pre_matches = np.insert(pre_matches, 4, 1, axis=1)

        unmatched_tracks = appearance_results[1]
        unmatched_detections = appearance_results[2]

    return pre_matches, unmatched_tracks, unmatched_detections


def better_np_unique(arr):

    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, _, counts = np.unique(arr,
                                               return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes, counts


def two_round_match(conflicts, alpha_gate):

    if conflicts.shape[0] == 0:
        return [], [], []

    first_round = conflicts[conflicts[:, 2] >= alpha_gate, :]
    matches_a = first_round[first_round[:, 3] == 1, :]
    if len(matches_a) != 0:
        matches_a[:, 4] = 2
    second_round = conflicts[conflicts[:, 2] < alpha_gate, :]
    second_round = second_round[second_round[:, 3] == 0, :]
    if len(second_round) != 0:
        second_round[:, 4] = 3
    second_round = select_pairs(second_round, 0, matches_a[:, 0])

    second_round = select_pairs(second_round, 1, matches_a[:, 1])

    matches_b = second_round[:, :]

    matches_ = np.vstack((matches_a, matches_b))
    matches_ = matches_.astype(np.int_)
    unmatched_tracks_ = list(set(conflicts[:, 0]) - set(matches_[:, 0]))
    unmatched_tracks_ = [int(i) for i in unmatched_tracks_]
    unmatched_detections_ = list(set(conflicts[:, 1]) - set(matches_[:, 1]))
    unmatched_detections_ = [int(i) for i in unmatched_detections_]

    return list(map(tuple, matches_)), unmatched_tracks_, unmatched_detections_


def filter_pairs(cost_matrix, gate):
    cost_matrix[cost_matrix > gate] = INFTY_COST

    return cost_matrix


def select_pairs(matches, col_id, pairs):

    for t in pairs:
        ind = np.argwhere(matches[:, col_id] == t)
        if len(ind):
            matches[ind[0], 3] = -1
    matches = matches[matches[:, 3] == 0, :]

    return matches
