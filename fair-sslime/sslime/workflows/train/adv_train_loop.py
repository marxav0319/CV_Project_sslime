#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from tqdm import tqdm

import torch
import torch.nn as nn


from sslime.core.config import config as cfg
from sslime.meters import METERS
from sslime.utils.utils import is_log_iter, log_train_stats

logger = logging.getLogger(__name__)


def adv_train_loop(train_loader, model, criterion, optimizer, scheduler, i_epoch):
    model.train()
    train_meters = [
        METERS[meter](**cfg.TRAINER.TRAIN_METERS[meter])
        for meter in cfg.TRAINER.TRAIN_METERS
    ]

    num_steps = 5
    epsilon = 0.001
    data_low = 0
    data_up = 1

    for i_batch, batch in enumerate(tqdm(train_loader)):
        batch["data"] = torch.cat(batch["data"]).cuda()
        batch["label"] = torch.cat(batch["label"]).cuda()


        noise = torch.empty(batch["data"].shape).cuda()
        nn.init.uniform_(noise, -epsilon, epsilon)
        x = batch["data"] + noise  # Random start
        x = torch.clamp(x, data_low, data_up) # x must remain in its domain
        # Might need x.requires_grad = true
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



        out = model(x)
        loss = criterion(out, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for meter in train_meters:
                meter.update(out, batch["label"])

        if is_log_iter(i_batch):
            log_train_stats(
                i_epoch, i_batch, len(train_loader), optimizer, train_meters
            )

    scheduler.step()
    logger.info(f"Epoch: {i_epoch + 1}. Train Stats")
    for meter in train_meters:
        logger.info(meter)

