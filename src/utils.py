import os
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from datetime import datetime, timezone, timedelta


def get_timestamp():
    timestamp = datetime.now(tz=timezone(timedelta(hours=9), "JST")).strftime(
        "%Y%m%d_%H%M%S"
    )
    return timestamp


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_folds(train, mode="predefined", random_state=0):
    if mode == "predefined":
        return PredefinedSplit(
            train.chip_id.map(
                {
                    "0b9dbf13f938efd5717f": 0,
                    "118c70535bd753a86615": 1,
                    "6718e7f83c824b1e436d": 2,
                    "79ad4647da6de6425abf": 3,
                    "84b788fdc5e779f8a0df": 4,
                    "c695a1e61e002b34e556": -1,
                }
            )
        )
    elif mode == "stratified":
        return StratifiedKFold(5, shuffle=True, random_state=random_state)


def prauc(y_true, y_pred):
    """Precision-Recall AUC for LightGBM"""
    score = average_precision_score(y_true, y_pred)
    return "prauc", score, True
