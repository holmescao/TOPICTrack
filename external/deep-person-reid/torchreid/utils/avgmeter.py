from __future__ import division, absolute_import
from collections import defaultdict
import torch

__all__ = ["AverageMeter", "MetricMeter"]


class AverageMeter(object):


    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):


    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError("Input to MetricMeter.update() must be a dictionary")

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append("{} {:.4f} ({:.4f})".format(name, meter.val, meter.avg))
        return self.delimiter.join(output_str)
