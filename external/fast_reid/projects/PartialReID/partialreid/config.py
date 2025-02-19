

from fast_reid.fastreid.config import CfgNode as CN


def add_partialreid_config(cfg):
    _C = cfg

    _C.TEST.DSR = CN({"ENABLED": True})
