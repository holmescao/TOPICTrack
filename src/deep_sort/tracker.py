import time
import numpy as np
import torch

from src.deep_sort.utils.parser import get_config
from src.deep_sort.deep_sort import DeepSort


cfg = get_config()
cfg.merge_from_file(
    "configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def step(results, RESET, opt):
   
    bboxes_xyxy = []
    previous_cts = []
    cts = []
    confs = []
    point_embeddings = []

    if len(results) == 0:
  
        pass

    start_time = time.time()
    for ct, previous_ct, box, conf, embedding in extract_det_info(results):
        bboxes_xyxy.append(box)
        cts.append(ct)
        previous_cts.append(previous_ct)
        confs.append(conf)
        point_embeddings.append(embedding)

    bboxes_xyxy = torch.Tensor(bboxes_xyxy)
    cts = torch.Tensor(cts)
    previous_cts = torch.Tensor(previous_cts)
    confss = torch.Tensor(confs)
    embeddings = torch.Tensor(point_embeddings)

    extract_info_time = time.time()
    extract_t = extract_info_time - start_time
    if RESET == True:
        deepsort.max_age = opt.max_age
        deepsort.reset()
    outputs = deepsort.update(opt,
                              bboxes_xyxy, cts, previous_cts, confss, embeddings)

    update_time = time.time()
    update_t = update_time - extract_info_time

    results = []
    tracks = []
    for value in list(outputs):
        x1, y1, x2, y2, track_id, alpha = value
        results.append({"bbox": (x1, y1, x2, y2),
                        "tracking_id": int(track_id),
                       "alpha": float(alpha)
                        })
        tracks.append({"bbox": np.array([x1, y1, x2, y2], dtype=np.float32)})

    res_time = time.time()
    res_t = res_time - update_time
    track_time = time.time()
    track_t = track_time - start_time
  

    return results, tracks


def extract_det_info(results):
    det_boxes = np.array([[item['bbox'][0], item['bbox'][1],
                           item['bbox'][2], item['bbox'][3]] for item in results], np.float32)   

    ct = np.array(
        [det['ct'] for det in results], np.float32) 
    previous_ct = np.array(
        [det['ct'] + det['tracking'] for det in results], np.float32) 

    item_score = np.array([item['score']
                           for item in results], np.float32) 
    item_embedding = np.array([item['embedding']
                               for item in results], np.float32) 

    return zip(ct, previous_ct, det_boxes, item_score, item_embedding)
