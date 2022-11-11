import logging
import sys, os, random
from typing import Dict, List

sys.path.append("../..")
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ResourceWarning)

from lib.pyclee.pyclee.forgetting import ForgettingMethod
from utils import read_config_file, run_dyclee, read_csv
from prediction_strategy.dynamo import DynAmo
from eval.metrics import Metric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import optuna
from optuna.visualization import plot_contour, plot_edf, plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice  

from os import listdir
from os.path import isfile, join
import datetime

# Seed value
seed_value= 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

def experiment(
    drift_detection_threshold: float,
    lookup_size=365,
    window_moving_step=2,
    limit_per_window=30,
    plot: bool = False,
    store: bool = False) -> List[float]:

    performances: Dict[str, list] = dict()

    for k, filepath in enumerate(DATA_PATH):
        res = read_config_file(CONFIG_FILEPATH, filepath)

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

        if plot:

            pred_drift_indices = [i + dynamo.lookup_size for i, e in enumerate(y_pred) if e == 1]
            true_drift_indices = np.where(y_true == 1)[0]

            fig, axes = plt.subplots(signal.shape[1], 1, figsize=(10,2*signal.shape[1]))
            fig.suptitle(f"\nHistory lookup: {dynamo.lookup_size}, Threshold: {round(dynamo.drift_detection_threshold, 4)}", y=0.95)

            for i in range(signal.shape[1]):
                axes[i].plot(signal.values[:,i])
                for a in true_drift_indices:
                    axes[i].axvline(x = a, linewidth=1, color='red', alpha=0.25)
                for j in pred_drift_indices:
                    axes[i].axvline(x = j, linewidth=1, color='g', alpha=0.15)

            plt.show()

        for metric in eval_metrics:
            if metric.name in performances:
                performances[metric.name].append(metric.calculate(y_true, y_pred))
            else:
                performances[metric.name] = [metric.calculate(y_true, y_pred)]

        if store:
            now = datetime.datetime.now()
            storage_filename = f'../../res/experiments/24-10-2022/others/average-voting/{now}-{filepath[filepath.rindex("/")+1:filepath.rindex(".")]}'\
                    +f'-lookup_size={dynamo.lookup_size}-wnd_moving_step={dynamo.wnd_moving_step}'\
                        +f'-drift_detection_threshold={dynamo.drift_detection_threshold}'\
                            +f'-limit_per_window={dynamo.limit_per_window}.csv'

            res = pd.DataFrame(np.array(list(performances.values()))[:,k].T,\
                index=[x.name for x in eval_metrics])
            res.to_csv(storage_filename)

        print(performances.items())

    for metric in eval_metrics:
        performances[metric.name] = np.mean(performances[metric.name])

    return performances["F1Score"]


def objective(trial):
    th = trial.suggest_float("drift_detection_threshold", 0.05, 0.95)
    lookup_size = trial.suggest_int("lookup_size", 1, 20)
    limit_per_window = trial.suggest_int('limit_per_window', 4, 20)
    window_moving_step = trial.suggest_int('window_moving_step', 1, 10)

    for step in range(1):
        f = experiment(drift_detection_threshold=th, lookup_size=lookup_size, window_moving_step=window_moving_step,limit_per_window=limit_per_window)

        trial.report(f, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return f


if __name__ == '__main__':
    CONFIG_FILEPATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    DATA_PATH = [join(DATA_PATH, f) for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and f.endswith('.csv')]

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="maximize", sampler=optuna.integration.BoTorchSampler(seed=seed_value))
    study.optimize(objective, n_trials=100, timeout=10000)

    print(f'Best params = {study.best_params}')

    plot_optimization_history(study).write_image('optimization_history.svg')
    plot_parallel_coordinate(study).write_image('parallel_coordinate.svg')
    plot_contour(study).write_image('contour.svg')
    plot_slice(study).write_image('slice.svg')
    plot_param_importances(study).write_image('param_importances.svg')
    plot_edf(study).write_image('edf.svg')

