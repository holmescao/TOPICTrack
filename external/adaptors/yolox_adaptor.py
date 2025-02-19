import sys
import warnings

import torch
import torch.nn as nn

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import postprocess, fuse_model


class PostModel(nn.Module):
    def __init__(self, model, exp):
        super().__init__()
        self.exp = exp
        self.model = model

    def forward(self, batch):
        """
        Returns Nx5, (x1, y1, x2, y2, conf)
        """
        raw = self.model(batch)
        pred = postprocess(
            raw, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre
        )[0]
        if pred is not None:
            return torch.cat((pred[:, :4], (pred[:, 4] * pred[:, 5])[:, None]), dim=1)
        else:
            return None


def get_model(path, dataset):
    exp = Exp(dataset)
    model = exp.get_model()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    with warnings.catch_warnings():
        model = fuse_model(model)
    model = model.half()
    model = PostModel(model, exp)
    model.cuda()
    model.eval()
    return model


class Exp:
    def __init__(self, dataset):
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        if dataset == "mot17" or dataset == "dance":
            self.input_size = (800, 1440)
            self.test_size = (800, 1440)
        elif dataset == "mot20":
            self.input_size = (896, 1600)
            self.test_size = (896, 1600)
        elif dataset == "dancetrack":
            self.input_size = (896, 1600)
            self.test_size = (896, 1600)

        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
