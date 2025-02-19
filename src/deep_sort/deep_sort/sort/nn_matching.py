
from matplotlib.pyplot import yscale
import numpy as np
import torch
import torch.nn.functional as F

def _pdist(a, b):
   
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
   
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
  
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
  
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


def _nn_recons_cosine_distance(x, y):
   
    ftrk = torch.from_numpy(np.asarray(x)).cuda() 
    fdet = torch.from_numpy(np.asarray(y)).cuda() 
    
    aff = torch.mm(ftrk, fdet.transpose(0, 1))
 
    aff_td = F.softmax(aff, dim=1) 
    aff_dt = F.softmax(aff, dim=0).transpose(0, 1) 
    
    recons_ftrk = torch.mm(aff_td, fdet) 
    recons_fdet = torch.mm(aff_dt, ftrk) 
    recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    recons_fdet_norm = F.normalize(recons_fdet, dim=1)
    
    dot_td = torch.mm(ftrk, recons_ftrk_norm.transpose(0, 1))
    dot_dt = torch.mm(fdet, recons_fdet_norm.transpose(0, 1))
    distances = 0.5 * (dot_td + dot_dt.transpose(0, 1))
    distances = distances.detach().cpu().numpy()
    
    return distances.min(axis=0)

def _nn_res_recons_cosine_distance(x, y):
    
    
    ftrk = torch.from_numpy(np.asarray(x)).cuda() 
    fdet = torch.from_numpy(np.asarray(y)).cuda() 
  
    aff = torch.mm(ftrk, fdet.transpose(0, 1)) 
   
    aff_td = F.softmax(aff, dim=1)
    aff_dt = F.softmax(aff, dim=0).transpose(0, 1)
    
    res_recons_ftrk = torch.mm(aff_td, fdet) 
    res_recons_fdet = torch.mm(aff_dt, ftrk) 
    recons_ftrk = ftrk + res_recons_ftrk
    recons_fdet = fdet + res_recons_fdet
    
    recons_ftrk_norm = recons_ftrk
    recons_fdet_norm = recons_fdet
    
    distances = 1 - torch.mm(recons_ftrk_norm, recons_fdet_norm.transpose(0, 1))
    distances = distances.detach().cpu().numpy()
    
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    

    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        elif metric == "recons":
            self._metric = _nn_recons_cosine_distance
        elif metric == "res_recons":
            self._metric = _nn_res_recons_cosine_distance
        
        else:
            raise ValueError(
                "Invalid metric; must be 'euclidean', 'cosine', or 'recons'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
       
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
       
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
    
    

