from __future__ import print_function, absolute_import
import warnings
import torch
import torch.nn as nn

from .radam import RAdam

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam"]


def build_optimizer(
    model,
    optim="adam",
    lr=0.0003,
    weight_decay=5e-04,
    momentum=0.9,
    sgd_dampening=0,
    sgd_nesterov=False,
    rmsprop_alpha=0.99,
    adam_beta1=0.9,
    adam_beta2=0.99,
    staged_lr=False,
    new_layers="",
    base_lr_mult=0.1,
):
    if optim not in AVAI_OPTIMS:
        raise ValueError(
            "Unsupported optim: {}. Must be one of {}".format(optim, AVAI_OPTIMS)
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model given to build_optimizer must be an instance of nn.Module"
        )

    if staged_lr:
        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn("new_layers is empty, therefore, staged_lr is useless")
            new_layers = [new_layers]

        if isinstance(model, nn.DataParallel):
            model = model.module

        base_params = []
        base_layers = []
        new_params = []

        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        param_groups = [
            {"params": base_params, "lr": lr * base_lr_mult},
            {"params": new_params},
        ]

    else:
        param_groups = model.parameters()

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    return optimizer
