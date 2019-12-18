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

from sslime.core.config import config as cfg
from sslime.meters import METERS
from sslime.utils.utils import is_log_iter, log_train_stats

logger = logging.getLogger(__name__)


def empty_train_loop(train_loader, model, criterion, optimizer, scheduler, i_epoch):
    model.train()
    
