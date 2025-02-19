

import torch
from ray import tune

from fast_reid.fastreid.engine.hooks import EvalHook, flatten_results_dict
from fast_reid.fastreid.utils.checkpoint import Checkpointer


class TuneReportHook(EvalHook):
    def __init__(self, eval_period, eval_function):
        super().__init__(eval_period, eval_function)
        self.step = 0

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    )

        torch.cuda.empty_cache()

        self.step += 1

        with tune.checkpoint_dir(step=self.step) as checkpoint_dir:
            additional_state = {"epoch": int(self.trainer.epoch)}

            self.trainer.checkpointer.save_dir = checkpoint_dir
            self.trainer.checkpointer.save(name="checkpoint", **additional_state)

        metrics = dict(r1=results["Rank-1"], map=results["mAP"], score=(results["Rank-1"] + results["mAP"]) / 2)
        tune.report(**metrics)
