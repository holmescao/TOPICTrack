

import megengine.module as M

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(M.Module):
    

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
           
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x):
    
        fpn_outs = self.backbone(x)
        assert not self.training
        outputs = self.head(fpn_outs)

        return outputs
