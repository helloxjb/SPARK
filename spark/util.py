import logging
import datetime
import logging
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def extract_features(model, loader):
    targets, features = [], []

    state = model.training
    model.eval()

    for input_dict in loader:
        inputs, _targets = input_dict["inputs"], input_dict["targets"]

        _targets = _targets.numpy()
        _features = model.extract(inputs.to(model.device)).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    model.train(state)

    return np.concatenate(features), np.concatenate(targets)