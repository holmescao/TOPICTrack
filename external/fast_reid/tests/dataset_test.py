

import sys
sys.path.append('.')
from data import get_dataloader
from config import cfg
import argparse
from data.datasets import init_dataset

cfg.DATASETS.NAMES = ("market1501", "dukemtmc", "cuhk03", "msmt17", "mot17", "mot20",)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        '-cfg', "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    get_dataloader(cfg)

