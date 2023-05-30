import logging
import os
import random
import sys
from typing import Dict, List

sys.path.append("../..")
sys.path.append("..")
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ResourceWarning)
import argparse
import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
from optuna.visualization import (plot_contour, plot_edf,
                                  plot_optimization_history,
                                  plot_parallel_coordinate,
                                  plot_param_importances, plot_slice)

from eval.metrics import Metric
from lib.pyclee.forgetting import ForgettingMethod
from prediction_strategy.dynamo import DynAmo
from utils import read_config_file, run_dyclee


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
            dataset_name = filepath[:filepath.rindex(os.pathsep)]
            dataset_name = dataset_name[dataset_name.rindex(os.pathsep) + 1:]
            filename = filename[filepath.rindex(os.pathsep) + 1:filepath.rindex('.')]
            
            storage_filename = f"{join('/res', 'experiments', now.date, dataset_name, dynamo.consensus_func.name)}"\
                + f"{filename}-lookup_size={dynamo.lookup_size}-wnd_moving_step={dynamo.wnd_moving_step}"\
                    + f"-drift_detection_threshold={dynamo.drift_detection_threshold}"\
                        + f"-limit_per_window={dynamo.limit_per_window}.csv"

            res = pd.DataFrame(np.array(list(performances.values()))[:,k].T,\
                index=[x.name for x in eval_metrics])
            res.to_csv(storage_filename)

    for metric in eval_metrics:
        performances[metric.name] = np.mean(performances[metric.name])

    return performances["F1Score"]


def objective(trial):
    th = trial.suggest_float("drift_detection_threshold", TH_L, TH_U)
    lookup_size = trial.suggest_int("lookup_size", int(LOOKUP_L), int(LOOKUP_U))
    limit_per_window = trial.suggest_int('limit_per_window', int(WND_L), int(WND_U))
    window_moving_step = trial.suggest_int('window_moving_step', int(MV_STEP_L), int(MV_STEP_U))

    for step in range(TRIAL_STEPS):
        f = experiment(drift_detection_threshold=th,
                       lookup_size=lookup_size,
                       window_moving_step=window_moving_step,
                       limit_per_window=limit_per_window,
                       plot=PLOT,
                       store=STORE)

        trial.report(f, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return f


def list_of_numbers(string):
    try:
        numbers = [float(num) for num in string.split(',')]
        if len(numbers) != 2:
            raise argparse.ArgumentTypeError('The length of the list must be 2')
        if numbers[0] > numbers[1]:
            raise argparse.ArgumentTypeError(f'The first number needs to be < than the second\n'\
                +f'You provided {numbers[0]} and {numbers[1]}')
        return numbers
    except ValueError:
        raise argparse.ArgumentTypeError(f'Invalid list of numbers {string}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("config_file",
                        help="Configuration file for DynAmo")
    
    parser.add_argument("data_path",
                        help="Directory that contains the dataset(s)")
    
    parser.add_argument("--trial_num",
                        type=int,
                        help="Number of trials for optimisation",
                        default=100)
    
    parser.add_argument("--timeout",
                        type=int,
                        help="Timeout after which the optimisation stops",
                        default=10000)
    
    parser.add_argument("--trial_steps",
                        type=int,
                        help="How many optimisation trial steps do you want to do?",
                        default=1)
    
    parser.add_argument("--plot",
                        action='store_true',
                        help="Do you want to plot the optimisation history?")
    
    parser.add_argument("--store",
                        action='store_true',
                        help="Do you want to store the the plots and the optimised metrics dataframe?")
    
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Seed for reproducibility")
    
    parser.add_argument("--drift_detection_threshold",
                        type=list_of_numbers,
                        default=[0.05, 0.95],
                        help="Lower and upper bound (comma separated) for σ")
    
    parser.add_argument("--lookup_size",
                        type=list_of_numbers,
                        default=[1, 20],
                        help="Lower and upper bound (comma separated) for λ")
    
    parser.add_argument("--limit_per_window",
                        type=list_of_numbers,
                        default=[4, 30],
                        help="Lower and upper bound (comma separated) for ℓ")
    
    parser.add_argument("--window_moving_step",
                        type=list_of_numbers,
                        default=[1, 10],
                        help="Lower and upper bound (comma separated) for δ")
    
    args = parser.parse_args()
    
    CONFIG_FILEPATH = args.config_file
    DATA_PATH = args.data_path
    DATA_PATH = [join(DATA_PATH, f) for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and f.endswith('.csv')]
    TH_L, TH_U = args.drift_detection_threshold
    LOOKUP_L, LOOKUP_U = args.lookup_size
    WND_L, WND_U = args.limit_per_window
    MV_STEP_L, MV_STEP_U = args.window_moving_step
    TRIAL_STEPS = args.trial_steps
    STORE = args.store
    PLOT = args.plot
    
    seed_value = args.seed
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="maximize", sampler=optuna.integration.BoTorchSampler(seed=seed_value))
    study.optimize(objective, n_trials=args.trial_num, timeout=args.timeout)

    if args.store:
        plot_optimization_history(study).write_image('optimization_history.svg')
        plot_parallel_coordinate(study).write_image('parallel_coordinate.svg')
        plot_contour(study).write_image('contour.svg')
        plot_slice(study).write_image('slice.svg')
        plot_param_importances(study).write_image('param_importances.svg')
        plot_edf(study).write_image('edf.svg')

