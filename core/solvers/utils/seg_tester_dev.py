# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass

