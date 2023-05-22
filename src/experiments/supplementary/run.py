import os
import random
import datetime
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from lib.pyclee.pyclee.forgetting import ForgettingMethod

from eval.metrics import Metric
from prediction_strategy.dynamo import DynAmo
from utils import read_config_file, run_dyclee


def experiment(
    drift_detection_threshold: float,
    lookup_size=365,
    window_moving_step=2,
    limit_per_window=30,
    store=True):
    
    performances: Dict[str, list] = dict()

    res = read_config_file(CONFIG_FILEPATH, DATASET_PATH)

    signal: pd.DataFrame = res['signal']
    y: pd.DataFrame = res['y']
    hyperbox_fraction: float  = res['hyperbox_fraction']
    forgetting_method: ForgettingMethod = res['forgetting_instance']
    dynamo: DynAmo = res['dynamo']
    eval_metrics: List[Metric] = res['eval_strategy']


    new_signal: pd.DataFrame = run_dyclee(
        signal.values,
        hyperbox_fraction=hyperbox_fraction,
        forgetting_method=forgetting_method
    )

    dynamo.signal = new_signal
    dynamo.lookup_size = lookup_size
    dynamo.drift_detection_threshold = drift_detection_threshold
    dynamo.limit_per_window = limit_per_window
    dynamo.wnd_moving_step = window_moving_step
    y_pred: List[int] = dynamo.run()
    y_true = y.values

    for metric in eval_metrics:
        if metric.name in performances:
            performances[metric.name].append(metric.calculate(y_true, y_pred))
        else:
            performances[metric.name] = [metric.calculate(y_true, y_pred)]

    if store:
        now = datetime.datetime.now()
        storage_filename = f'../../res/experiments/24-10-2022/others/average-voting/{now}-{DATASET_PATH[:DATASET_PATH.rindex(".")]}'\
                +f'-lookup_size={dynamo.lookup_size}-wnd_moving_step={dynamo.wnd_moving_step}'\
                    +f'-drift_detection_threshold={dynamo.drift_detection_threshold}'\
                        +f'-limit_per_window={dynamo.limit_per_window}.csv'

        res = pd.DataFrame(np.array(list(performances.values())).T,\
            index=[x.name for x in eval_metrics])
        res.to_csv(storage_filename)

    for metric in eval_metrics:
        performances[metric.name] = np.mean(performances[metric.name])

    return performances
    
    
if __name__ == '__main__':

    CONFIG_FILEPATH = sys.argv[1]
    DATASET_PATH = sys.argv[2]

    drift_detection_threshold = np.random.rand(1,1)[0]
    lookup_size = np.random.randint(0, 365)
    window_moving_step = np.random.randint(1, 10)
    limit_per_window = np.random.randint(4,30)

    experiment(drift_detection_threshold=drift_detection_threshold,
            lookup_size=lookup_size,
            window_moving_step=window_moving_step,
            limit_per_window=limit_per_window)