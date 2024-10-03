from typing import Tuple
import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import torch
from sklearn.model_selection import train_test_split

from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import download_man_ahl_dataset, download_mit_ecg_dataset
from lib.data import get_data
from lib.plot import savefig, create_summary
from lib.utils import pickle_it, load_pickle
from lib.utils import sample_indices
from lib.test_metrics import ACFLoss
from lib.test_metrics import acf_torch, cacf_torch, lev_eff_torch


from pathos.multiprocessing import ProcessingPool as Pool


# calculate the metrics for the fake data
def calculate_metrics(fake_data: torch.Tensor, real_data: torch.Tensor) -> Tuple:
    # acf = acf_torch(fake_data, 64)
    # cacf = cacf_torch(fake_data, 64)
    # lev_eff = lev_eff_torch(fake_data, 20)
    # pacf = pacf_torch(fake_data, 100)
    # pacf2 = pacf2_torch(fake_data, 100)

    acf = ACFLoss(name=None, x_real=real_data, max_lag=64)
    acf_loss = acf(fake_data)

    # cacf_loss = CACFLoss()(cacf, real_data)
    # lev_loss = LEVLoss()(lev_eff, real_data)
    # pacf_loss = PACFLoss()(pacf, real_data)
    # pacf2_loss = PACFLoss2()(pacf2, real_data)

    return acf_loss,
    # cacf_loss, lev_loss, pacf_loss, pacf2_loss

