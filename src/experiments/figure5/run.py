import sys, os, random
from typing import Dict, List
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from lib.pyclee.pyclee.forgetting import ForgettingMethod
from utils import read_config_file, run_dyclee, read_csv
from prediction_strategy.dynamo import DynAmo
from eval.metrics import Metric
import pandas as pd
import numpy as np

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
    tau = int(sys.argv[4])

    res = read_config_file(config_filepath, input_data)
    signal, y = res['signal'], res['y']
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
    dynamo.lookup_size = tau
    y_pred: List[int] = dynamo.run()

    performances: Dict[str, float] = dict()
    for metric in eval_metrics:
        performances[metric.name] = metric.calculate(y, y_pred)

    performances_df = pd.DataFrame(np.array(list(performances.values())).reshape(1, len(performances)), columns=list(performances.keys()))
    performances_df.to_csv(report_metrics_filepath, index=False)