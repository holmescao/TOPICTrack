import time
import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.3, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap


        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric(
            "res_recons", 
            max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

    def reset(self):
        self.tracker = Tracker(
            self.metric, max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init)

    def update(self, opt, bbox_xyxy, cts, previous_cts, confidences, features):
       
        start_time = time.time()
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], cts[i], previous_cts[i], conf, features[i])
                      for i, conf in enumerate(confidences) if conf > self.min_confidence]
        det_time = time.time()
        det_t = det_time - start_time
        
        self.tracker.predict()
        KFpre_time = time.time()
        KFpre_t = KFpre_time - det_time
        self.tracker.update(detections, opt)
        update_time = time.time()
        update_t = update_time - KFpre_time

        
        outputs = np.zeros((200,6))
        for i, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = self._tlwh_to_xyxy(track.tlwh)
          
            alpha = 1

            outputs[i,:] = np.array([x1, y1, x2, y2, track.track_id, alpha],dtype=np.float32)
            
        outputs = outputs[outputs[:,-2] >0]

        out_time = time.time()
        out_t = out_time - update_time
        

        return outputs

   
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()

        bbox_tlwh[:, 0] = bbox_xyxy[:, 0]
        bbox_tlwh[:, 1] = bbox_xyxy[:, 1]
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]

        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = int(x+w/2)
      
        y1 = max(int(y-h/2), 0)
        y2 = int(y+h/2)
       
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        
        x, y, w, h = bbox_tlwh
        x1 = max(x, 0)
        x2 = x+w
       
        y1 = max(y, 0)
        y2 = y+h
     
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
