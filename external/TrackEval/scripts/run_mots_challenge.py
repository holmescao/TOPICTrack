

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import trackeval  

if __name__ == "__main__":
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = False
    default_dataset_config = (
        trackeval.datasets.MOTSChallenge.get_default_dataset_config()
    )
    default_metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
    config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    } 
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == "True":
                    x = True
                elif args[setting] == "False":
                    x = False
                else:
                    raise Exception(
                        "Command line parameter " + setting + "must be True or False"
                    )
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == "SEQ_INFO":
                x = dict(zip(args[setting], [None] * len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in config.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in config.items() if k in default_metrics_config.keys()
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MOTSChallenge(dataset_config)]
    metrics_list = []
    for metric in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
        trackeval.metrics.VACE,
        trackeval.metrics.JAndF,
    ]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")
    evaluator.evaluate(dataset_list, metrics_list)
