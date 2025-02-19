

import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode


_C = CfgNode()

cfg = _C

_C.MODEL = CfgNode()

_C.MODEL.TYPE = ""

_C.MODEL.DEPTH = 0

_C.MODEL.NUM_CLASSES = 10

_C.MODEL.LOSS_FUN = "cross_entropy"

_C.RESNET = CfgNode()

_C.RESNET.TRANS_FUN = "basic_transform"

_C.RESNET.NUM_GROUPS = 1

_C.RESNET.WIDTH_PER_GROUP = 64

_C.RESNET.STRIDE_1X1 = True

_C.ANYNET = CfgNode()

_C.ANYNET.STEM_TYPE = "simple_stem_in"

_C.ANYNET.STEM_W = 32

_C.ANYNET.BLOCK_TYPE = "res_bottleneck_block"

_C.ANYNET.DEPTHS = []

_C.ANYNET.WIDTHS = []

_C.ANYNET.STRIDES = []

_C.ANYNET.BOT_MULS = []

_C.ANYNET.GROUP_WS = []

_C.ANYNET.SE_ON = False

_C.ANYNET.SE_R = 0.25


_C.REGNET = CfgNode()

_C.REGNET.STEM_TYPE = "simple_stem_in"

_C.REGNET.STEM_W = 32

_C.REGNET.BLOCK_TYPE = "res_bottleneck_block"

_C.REGNET.STRIDE = 2

_C.REGNET.SE_ON = False
_C.REGNET.SE_R = 0.25

_C.REGNET.DEPTH = 10

_C.REGNET.W0 = 32

_C.REGNET.WA = 5.0

_C.REGNET.WM = 2.5

_C.REGNET.GROUP_W = 16

_C.REGNET.BOT_MUL = 1.0

_C.EN = CfgNode()

_C.EN.STEM_W = 32

_C.EN.DEPTHS = []

_C.EN.WIDTHS = []

_C.EN.EXP_RATIOS = []

_C.EN.SE_R = 0.25

_C.EN.STRIDES = []

_C.EN.KERNELS = []

_C.EN.HEAD_W = 1280

_C.EN.DC_RATIO = 0.0

_C.EN.DROPOUT_RATIO = 0.0

_C.BN = CfgNode()

_C.BN.EPS = 1e-5

_C.BN.MOM = 0.1

_C.BN.USE_PRECISE_STATS = True
_C.BN.NUM_SAMPLES_PRECISE = 8192

_C.BN.ZERO_INIT_FINAL_GAMMA = False

_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

_C.OPTIM = CfgNode()

_C.OPTIM.BASE_LR = 0.1

_C.OPTIM.LR_POLICY = "cos"

_C.OPTIM.GAMMA = 0.1

_C.OPTIM.STEPS = []

_C.OPTIM.LR_MULT = 0.1

_C.OPTIM.MAX_EPOCH = 200

_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.DAMPENING = 0.0

_C.OPTIM.NESTEROV = True

_C.OPTIM.WEIGHT_DECAY = 5e-4

_C.OPTIM.WARMUP_FACTOR = 0.1

_C.OPTIM.WARMUP_ITERS = 0

_C.TRAIN = CfgNode()

_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

_C.TRAIN.BATCH_SIZE = 128

_C.TRAIN.IM_SIZE = 224

_C.TRAIN.EVAL_PERIOD = 1

_C.TRAIN.CHECKPOINT_PERIOD = 1

_C.TRAIN.AUTO_RESUME = True

_C.TRAIN.WEIGHTS = ""

_C.TEST = CfgNode()

_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

_C.TEST.BATCH_SIZE = 200

_C.TEST.IM_SIZE = 256

_C.TEST.WEIGHTS = ""

_C.DATA_LOADER = CfgNode()

_C.DATA_LOADER.NUM_WORKERS = 8

_C.DATA_LOADER.PIN_MEMORY = True

_C.MEM = CfgNode()

_C.MEM.RELU_INPLACE = True

_C.CUDNN = CfgNode()

_C.CUDNN.BENCHMARK = True

_C.PREC_TIME = CfgNode()

_C.PREC_TIME.WARMUP_ITER = 3

_C.PREC_TIME.NUM_ITER = 30

_C.NUM_GPUS = 1

_C.OUT_DIR = "/tmp"

_C.CFG_DEST = "config.yaml"

_C.RNG_SEED = 1

_C.LOG_DEST = "stdout"

_C.LOG_PERIOD = 10

_C.DIST_BACKEND = "nccl"

_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]

_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"


_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")
_C.register_deprecated_key("PORT")


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)
