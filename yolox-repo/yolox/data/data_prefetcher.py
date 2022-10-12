#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)

    def next(self):
        input, target, _, _ = next(self.loader)
        return input, target
