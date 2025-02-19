from __future__ import division, print_function, absolute_import
import torch

from torchreid.engine.image import ImageTripletEngine


class VideoTripletEngine(ImageTripletEngine):

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        pooling_method="avg",
    ):
        super(VideoTripletEngine, self).__init__(
            datamanager,
            model,
            optimizer,
            margin=margin,
            weight_t=weight_t,
            weight_x=weight_x,
            scheduler=scheduler,
            use_gpu=use_gpu,
            label_smooth=label_smooth,
        )
        self.pooling_method = pooling_method

    def parse_data_for_train(self, data):
        imgs = data["img"]
        pids = data["pid"]
        if imgs.dim() == 5:
            
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b * s, c, h, w)
            pids = pids.view(b, 1).expand(b, s)
            pids = pids.contiguous().view(b * s)
        return imgs, pids

    def extract_features(self, input):
       
        b, s, c, h, w = input.size()
        input = input.view(b * s, c, h, w)
        features = self.model(input)
        features = features.view(b, s, -1)
        if self.pooling_method == "avg":
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
        return features
