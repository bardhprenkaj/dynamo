import os
import random
import sys
from typing import Dict, List

sys.path.append("..")

import warnings

import numpy as np
import pandas as pd

from eval.metrics import Metric
from lib.pyclee.forgetting import ForgettingMethod
from prediction_strategy.dynamo import DynAmo
from utils import read_config_file, run_dyclee

warnings.filterwarnings("ignore")

# Seed value
seed_value= 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

if __name__ == '__main__':

    config_filepath = sys.argv[1]
    input_data = sys.argv[2]
    report_metrics_filepath = sys.argv[3]

    res = read_config_file(config_filepath, input_data)
    signal, y = res['signal'], res['y']
    hyperbox_fraction: float  = res['hyperbox_fraction']
    forgetting_method: ForgettingMethod = res['forgetting_instance']
    dynamo: DynAmo = res['dynamo']
    eval_metrics: List[Metric] = res['eval_strategy']

    start_offset = res['start_offset']
    y_true = y.values[start_offset:]

    print(dynamo)

    new_signal: pd.DataFrame = run_dyclee(
        signal.values[start_offset:],
        hyperbox_fraction=hyperbox_fraction,
        forgetting_method=forgetting_method
    )

    dynamo.signal = new_signal
    y_pred: List[int] = dynamo.run()

    performances: Dict[str, float] = dict()
    for metric in eval_metrics:
        performances[metric.name] = metric.calculate(y_true, y_pred)

    performances_df = pd.DataFrame(np.array(list(performances.values())).reshape(1, len(performances)), columns=list(performances.keys()))
    performances_df.to_csv(report_metrics_filepath, index=False)