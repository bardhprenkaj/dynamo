import abc
import json
import sys
from pydoc import locate

sys.path.append("../..")
sys.path.append("..")
import os
import random

##################################################################################################################
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from itertools import count
##################################################################################################################
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.pyclee.clusters import Cluster
from lib.pyclee.dyclee import DyClee, DyCleeContext
from lib.pyclee.forgetting import ExponentialForgettingMethod, ForgettingMethod
from lib.pyclee.types import Element, Timestamp
from src.eval.metrics import Metric
from src.prediction_strategy.divergency.tests import (
    DensestHyperboxDifference, DivergenceMetric, MeanDivergence)
from src.prediction_strategy.dynamo import DynAmo
from src.prediction_strategy.ensemble.trackers import (
    BoxSizeProductTracker, BoxSizeTracker, DifferenceBoxTracker,
    NormalizedBoxSizeTracker, NormalizedDifferenceBoxTracker, Tracker)
from src.prediction_strategy.voting.consensus_functions import (Consensus,
                                                                MajorityVoting)

##################################################################################################################
# Seed value
seed_value= 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Normalizes each column of the data frame.

        Parameters
        ----------
        df : pd.DataFrame
            The data frame to normalize

        Returns
        -------
        pd.DataFrame
            the normalized column-wise data frame

        Raises
        ------
        ZeroDivisionError
            If the maximium and the minimum values of 
            a specific column coincide.
    """
    for i in range(len(df.columns)):
        temp = df.iloc[:,i:i+1]

        if temp.max()[0] != temp.min()[0]:  
            temp_norm = (temp - temp.mean()) / (temp.max() - temp.min())
        else:
            raise ZeroDivisionError(f'Column {df.columns[i]} has the same min = max = {temp.max()[0]}')

        df.iloc[:,i:i+1] = temp_norm

    return df

def read_csv(filepath: str, sep: str=',', label_col: int=-1, normalize: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Reads a csv file.

        Parameters
        ----------
        filepath : str
            the path to the file to read
        
        sep: str, optional
            the character used to separate the columns in the file (default is ',')

        label_col: int, optional
            the column that localizes where the label is in the csv file (default is -1)

        normalize: bool, optional
            flag that indicates whether to normalize the file read (default is True)

        start_offset: int, optional
            flag that indicates when to start in the trajectory

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            a tuple consisting of (data features, labels)
    """
    df = pd.read_csv(filepath, sep=sep)
    signal = df.iloc[:,:label_col]
    y = df.iloc[:,label_col]

    if normalize:
        signal = normalize_df(signal)
        
    return signal, y

def run_dyclee(
    signal: np.array,
    hyperbox_fraction: float=.2,
    forgetting_method: ForgettingMethod = None) -> pd.DataFrame:
    """ Runs the DyClee algorithm and finds the centroids of maximally densest clusters.

        Parameters
        ----------
        signal : np.array
            the input features
        
        hyperbox_fraction: float, optional
            Relative size of each dimension of the microclusters' hyperboxes, as a
            fraction of the total range of each dimension of the input data.
            If a scalar is given, the same fraction is used for all dimensions (default is .2)

        forgetting_method: lib.pyclee.pyclee.forgetting.ForgettingMethod, optional
            Function that will be applied to microclusters' element accumulators to
            "forget" older samples (as a function of time intervals). `None` implies
            unlimited temporal memory. (default is None)

        Returns
        -------
        np.array
            an array containing the centroids correpsonding to the densest clusters
    """
    def run_internal(
        dy: DyClee,
        elements: Iterable[Element],
        times: Optional[Iterable[Timestamp]] = None,
        progress: bool = True) -> List[Cluster]:
        
        if progress and tqdm is not None:
            elements = tqdm(elements)
        
        if times is None:
            times = count()
        
        for element, time in zip(elements, times):
            _, clusters, _, _ = dy.step(element, time, skip_density_step=False)
            all_clusters.append(clusters)
            
        dy.density_step(time)
        
        return all_clusters

    all_clusters: list[Cluster] = list()
    centroids: list[np.ndarray] = list()
    bounds = np.column_stack((signal.min(axis=0), signal.max(axis=0)))
    
    context = DyCleeContext(
        n_features=signal.shape[1],
        hyperbox_fractions=hyperbox_fraction,
        feature_ranges=bounds,
        forgetting_method=forgetting_method)

    dy = DyClee(context)

    all_clusters = run_internal(dy, signal)
    all_clusters = list(filter(lambda item: item is not None, all_clusters))


    for step in all_clusters:
        max_dense = 0
        for cluster in step:
            micros = cluster.μclusters
            density = sum([micro.n_elements for micro in micros])

            if density >= max_dense:

                max_dense = density
                centroid = cluster.centroid

        centroids.append(centroid)

    return np.array(centroids)
    
def read_config_file(filepath: str, input_data_path: str) -> Dict[str, object]:
    res: Dict[str, object] = dict()

    config = dict()
    with open(filepath, 'r') as f:
        config = json.load(f)

    ############################################################################################################################################################
    assert 'data' in  config.keys(), "'data' needs to be included in the configuration file"

    data_reader_configs = config.get('data', None)

    filepath = input_data_path

    normalize = data_reader_configs.get('normalize', None)
    normalize = (normalize.lower() == 'true') if normalize != None else False

    print(f'Normalize = {normalize}')

    label_col = data_reader_configs.get('label_column', None)
    label_col = label_col if label_col else -1

    sep = data_reader_configs.get('sep', None)
    sep = sep if sep else ','

    start_offset = data_reader_configs.get('start_offset', None)
    start_offset = start_offset if start_offset else 0

    signal, y = read_csv(
        filepath=filepath,
        sep = sep,
        normalize=normalize,
        label_col=label_col)

    res['signal'] = signal
    res['y'] = y
    res['start_offset'] = start_offset
    ############################################################################################################################################################
    assert 'dyclee' in data_reader_configs.keys(), "'dyclee' needs to be a key in 'data'"

    dyclee_data_configs = data_reader_configs.get('dyclee')

    hyperbox_fraction = dyclee_data_configs.get('hyperbox_fraction', None)
    hyperbox_fraction = hyperbox_fraction if hyperbox_fraction else .2

    forgetting_method = dyclee_data_configs.get('forgetting_method', None)
    if forgetting_method:
        assert isinstance(forgetting_method, dict), "'forgetting_method' needs to be a dictionary type"
        assert len(forgetting_method.keys()) == 1, "Specify only one forgetting method for DyClee"

        method = list(forgetting_method.keys())[0]
        forgetting_parameters = forgetting_method.get(method, None)
        
        assert len(forgetting_parameters.keys()) > 0, f"Specify the forgetting parameters for method {method}"

        forgetting_class = locate(method)
        assert forgetting_class.__class__ == ForgettingMethod.__class__, f"Couldn't locate the forgetting_class {method}"
        forgetting_instance = forgetting_class(**forgetting_parameters)
    else:
        forgetting_instance = ExponentialForgettingMethod(.02)

    res['hyperbox_fraction'] = hyperbox_fraction
    res['forgetting_instance'] = forgetting_instance
    ############################################################################################################################################################
    assert 'dynamo' in  config.keys(), "'dynamo' needs to be included in the configuration file"

    dynamo_configs = config.get('dynamo', None)
    if dynamo_configs:
        lookup_size = dynamo_configs.get('lookup_size', None)
        lookup_size = lookup_size if lookup_size else 30

        drift_detection_threshold = dynamo_configs.get('drift_detection_threshold', None)
        drift_detection_threshold = drift_detection_threshold if drift_detection_threshold else .01

        wnd_moving_step = dynamo_configs.get('wnd_moving_step', None)
        wnd_moving_step = wnd_moving_step if wnd_moving_step else 2

        limit_per_window = dynamo_configs.get('limit_per_window', None)
        limit_per_window = limit_per_window if limit_per_window else 30

        consensus_func_name = dynamo_configs.get('consensus_func', None)
        if consensus_func_name:
            assert isinstance(consensus_func_name, dict), "'consensus_func' needs to be a dictionary type"
            assert len(consensus_func_name.keys()) == 1, "Specify only one consensus method for DynAmo"

            method = list(consensus_func_name.keys())[0]
            consensus_func_parameters = consensus_func_name.get(method, None)

            assert isinstance(consensus_func_parameters, dict),\
                "Specify the parameters of the consensus function. If the method doesn't take any parameters, leave a blank dictionary {}"
            
            consensus_func = locate(method)
            assert consensus_func.__class__ == Consensus.__class__, f"Couldn't locate the consensus {method}"
            consensus_instance = consensus_func(**consensus_func_parameters)
        else:
            consensus_instance = MajorityVoting()

        trackers = dynamo_configs.get('trackers', None)
        trackers_list: List[Tracker] = list()
        if trackers:
            assert isinstance(trackers, list), "'trackers' needs to be a list type"
            assert len(trackers) > 0, "Specify at least one tracker for DynAmo"

            for tracker in trackers:
                tracker_class = locate(tracker)
                assert tracker_class.__class__ == Tracker.__class__, f"Couldn't locate the tracker {tracker}"
                tracker_instance = tracker_class()
                trackers_list.append(tracker_instance)
        else:
            trackers_list = [
                BoxSizeTracker,
                BoxSizeProductTracker,
                NormalizedBoxSizeTracker,
                DifferenceBoxTracker,
                NormalizedDifferenceBoxTracker
            ]

        divergence_metrics = dynamo_configs.get('divergence_metrics', None)
        divergence_metrics_list: List[DivergenceMetric] = list()
        if divergence_metrics:
            assert isinstance(divergence_metrics, list), "'divergence_metrics' needs to be a list type"
            assert len(divergence_metrics) > 0, "Specify at least one divergence metric for DynAmo"

            for divergence_metric in divergence_metrics:
                divergence_metrics_class = locate(divergence_metric)
                assert divergence_metrics_class.__class__ == DivergenceMetric.__class__, f"Couldn't locate the divergence metric {divergence_metric}"
                divergence_metric_instance = divergence_metrics_class()
                divergence_metrics_list.append(divergence_metric_instance)
        else:
            divergence_metric_instance = [
                DensestHyperboxDifference,
                MeanDivergence
            ]

        dynamo = DynAmo(
            signal=None,
            trackers=trackers_list,
            divergence_metrics=divergence_metrics_list,
            consensus_func=consensus_instance,
            lookup_size=lookup_size,
            wnd_moving_step=wnd_moving_step,
            drift_detection_threshold=drift_detection_threshold,
            limit_per_window=limit_per_window
        )

        res['dynamo'] = dynamo
        ############################################################################################################################################################
        assert 'eval_strategy' in  config.keys(), "'eval_strategy' needs to be included in the configuration file"

        eval_strategy_configs = config.get('eval_strategy', None)
        eval_strategy_list: List[Metric] = list()
        if eval_strategy_configs:
            assert isinstance(eval_strategy_configs, list), "'eval_strategy' needs to be of type list"
            assert len(eval_strategy_configs) > 0, "Provide at least one evaluation metric for the 'eval_strategy'"

            for eval_metric in eval_strategy_configs:
                assert isinstance(eval_metric, dict), "Every evaluation metric needs to be of type dict"
                assert len(eval_metric.keys()) == 1, "The evaluation metric dicts need to have a single key"

                eval_metric_key = list(eval_metric.keys())[0]
                eval_metric_location = locate(eval_metric_key)
                assert eval_metric_location.__class__ == Metric.__class__, f"Couldn't locate the evaluation metric {eval_metric_key}"
                eval_metric_instance = eval_metric_location()
                eval_strategy_list.append(eval_metric_instance)

        res['eval_strategy'] = eval_strategy_list

        return res