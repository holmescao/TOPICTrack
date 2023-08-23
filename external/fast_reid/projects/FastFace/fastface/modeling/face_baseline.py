

import torch
from fast_reid.fastreid.modeling.meta_arch import Baseline
from fast_reid.fastreid.modeling.meta_arch import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class FaceBaseline(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pfc_enabled = cfg.MODEL.HEADS.PFC.ENABLED
        self.amp_enabled = cfg.SOLVER.AMP.ENABLED

    def forward(self, batched_inputs):
        if not self.pfc_enabled:
            return super().forward(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        with torch.cuda.amp.autocast(self.amp_enabled):
            features = self.backbone(images)
        features = features.float() if self.amp_enabled else features

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            return outputs, targets
        else:
            outputs = self.heads(features)
            return outputs