

import itertools

import torch

BN_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


@torch.no_grad()
def update_bn_stats(model, data_loader, num_iters: int = 200):
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    Args:
        model (nn.Module): the model whose bn stats will be recomputed.
            Note that:
            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.
            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        num_iters (int): number of iterations to compute the stats.
    """
    bn_layers = get_bn_modules(model)
    if len(bn_layers) == 0:
        return

    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    for ind, inputs in enumerate(itertools.islice(data_loader, num_iters)):
        inputs['targets'].fill_(-1)
        with torch.no_grad():  
            model(inputs)
        for i, bn in enumerate(bn_layers):
          
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)
           
    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
  
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]


def get_bn_modules(model):
   
    bn_layers = [
        m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)
    ]
    return bn_layers
