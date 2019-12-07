#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch
import torch.nn as nn

from sslime.core.config import config as cfg
from sslime.meters import METERS


logger = logging.getLogger(__name__)


def adv_eval_loop(val_loader, model, i_epoch):
    model.eval()
    eval_meters = [
        METERS[meter](**cfg.TRAINER.EVAL_METERS[meter])
        for meter in cfg.TRAINER.EVAL_METERS
    ]

    num_steps = 5
    epsilon = 0.001
    data_low = 0
    data_up = 1


    for batch in val_loader:
        batch["data"] = torch.cat(batch["data"]).cuda()
        batch["label"] = torch.cat(batch["label"]).cuda()


    noise = torch.empty(batch["data"].shape).cuda()
    nn.init.uniform_(noise, -epsilon, epsilon)
    x = batch["data"] + noise  # Random start
    x = torch.clamp(x, data_low, data_up) # x must remain in its domain
    x.requires_grad = True

    for i in range(num_steps):
        model.zero_grad()

        out = model(x)
        loss = criterion(out, batch["label"])
        loss.backward()

        x += step_size* x.grad.sign()
        x = torch.clamp(x, data_low, data_up)
        x = torch.clamp(x, batch["data"]-epsilon, batch["data"]+epsilon)
        x.grad.data.zero_()

    with torch.no_grad():
        out = model(x)
        for meter in eval_meters:
            meter.update(out, batch["label"])

    logger.info("Epoch: {}. Validation Stats".format(i_epoch + 1))
    for meter in eval_meters:
        logger.info(meter)
