

import logging
import os
import sys
from functools import partial

import ConfigSpace as CS
import ray
from hyperopt import hp
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch

sys.path.append('.')

from fast_reid.fastreid.config import get_cfg, CfgNode
from fast_reid.fastreid.engine import hooks
from fast_reid.fastreid.modeling import build_model
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fast_reid.fastreid.utils.events import CommonMetricPrinter
from fast_reid.fastreid.utils import comm
from fast_reid.fastreid.utils.file_io import PathManager

from autotuner import *

logger = logging.getLogger("fastreid.auto_tuner")

ray.init(dashboard_host='127.0.0.1')


class AutoTuner(DefaultTrainer):
    def build_hooks(self):
        r"""
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        ret.append(hooks.LayerFreeze(
            self.model,
            cfg.MODEL.FREEZE_LAYERS,
            cfg.SOLVER.FREEZE_ITERS,
            cfg.SOLVER.FREEZE_FC_ITERS,
        ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results


        ret.append(TuneReportHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():

            ret.append(hooks.PeriodicWriter([CommonMetricPrinter(self.max_iter)], 200))

        return ret

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def update_config(cfg, config):
    frozen = cfg.is_frozen()
    cfg.defrost()

    cfg.SOLVER.IMS_PER_BATCH = config["bsz"]
    cfg.DATALOADER.NUM_INSTANCE = config["num_inst"]

    if frozen: cfg.freeze()

    return cfg


def train_tuner(config, checkpoint_dir=None, cfg=None):
    update_config(cfg, config)

    tuner = AutoTuner(cfg)

    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint.pth")
        checkpoint = tuner.checkpointer.resume_or_load(path, resume=False)
        tuner.start_epoch = checkpoint.get("epoch", -1) + 1

    tuner.train()


def main(args):
    cfg = setup(args)

    exp_metrics = dict(metric="score", mode="max")

    if args.srch_algo == "hyperopt":
     
        search_space = {
            
            "bsz": hp.choice("bsz", [64, 96, 128, 160, 224, 256]),
            "num_inst": hp.choice("num_inst", [2, 4, 8, 16, 32]),
            
        }

        current_best_params = [{
            "bsz": 0,  
            "num_inst": 3,
        }]

        search_algo = HyperOptSearch(
            search_space,
            points_to_evaluate=current_best_params,
            **exp_metrics)

        if args.pbt:
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                **exp_metrics,
                perturbation_interval=2,
                hyperparam_mutations={
                    "bsz": [64, 96, 128, 160, 224, 256],
                    "num_inst": [2, 4, 8, 16, 32],
                }
            )
        else:
            scheduler = ASHAScheduler(
                grace_period=2,
                reduction_factor=3,
                max_t=7,
                **exp_metrics)

    elif args.srch_algo == "bohb":
        search_space = CS.ConfigurationSpace()
        search_space.add_hyperparameters([
           
            CS.CategoricalHyperparameter(name="bsz", choices=[64, 96, 128, 160, 224, 256]),
            CS.CategoricalHyperparameter(name="num_inst", choices=[2, 4, 8, 16, 32]),
            
        ])

        search_algo = TuneBOHB(
            search_space, max_concurrent=4, **exp_metrics)

        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            reduction_factor=3,
            max_t=7,
            **exp_metrics,
        )

    else:
        raise ValueError("Search algorithm must be chosen from [hyperopt, bohb], but got {}".format(args.srch_algo))

    reporter = CLIReporter(
        parameter_columns=["bsz", "num_inst"],
        metric_columns=["r1", "map", "training_iteration"])

    analysis = tune.run(
        partial(
            train_tuner,
            cfg=cfg),
        resources_per_trial={"cpu": 4, "gpu": 1},
        search_alg=search_algo,
        num_samples=args.num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=cfg.OUTPUT_DIR,
        keep_checkpoints_num=10,
        name=args.srch_algo)

    best_trial = analysis.get_best_trial("score", "max", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation mAP: {}, Rank-1: {}".format(
        best_trial.last_result["map"], best_trial.last_result["r1"]))

    save_dict = dict(R1=best_trial.last_result["r1"].item(), mAP=best_trial.last_result["map"].item())
    save_dict.update(best_trial.config)
    path = os.path.join(cfg.OUTPUT_DIR, "best_config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(CfgNode(save_dict).dump())
    logger.info("Best config saved to {}".format(os.path.abspath(path)))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--num-trials", type=int, default=8, help="number of tune trials")
    parser.add_argument("--srch-algo", type=str, default="hyperopt",
                        help="search algorithms for hyperparameters search space")
    parser.add_argument("--pbt", action="store_true", help="use population based training")
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
