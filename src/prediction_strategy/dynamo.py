import time
from typing import Dict, List, Tuple
import numpy as np

from prediction_strategy.divergency.tests import DivergenceMetric
from prediction_strategy.ensemble.trackers import Tracker
from prediction_strategy.voting.consensus_functions import Consensus, MajorityVoting

class DynAmo:

    def __init__(self,
        signal: np.array,
        trackers: List[Tracker],
        divergence_metrics: List[DivergenceMetric],
        consensus_func: Consensus = MajorityVoting,
        lookup_size: int = 30,
        drift_detection_threshold: float = .5,
        wnd_moving_step: int = 1,
        limit_per_window: int = 5):

        self.signal = signal
        self.trackers = trackers
        self.divergence_metrics = divergence_metrics
        self.consensus_func = consensus_func
        self.lookup_size = lookup_size
        self.drift_detection_threshold = drift_detection_threshold
        self.wnd_moving_step = wnd_moving_step
        self.limit_per_window = limit_per_window

        self.insert_reference_window: int = 0
        self.insert_detection_window: int = 0

        self.ensemble_voting: Dict[str, int] = dict()
        
    def __calc_box(self, x: np.array) -> np.array:
        return np.column_stack((x.min(axis=0), x.max(axis=0)))

    def __init_windows(self) -> Tuple[Dict[str, list], Dict[str, list]]:
        ref_wnd: Dict[str, list] = dict()
        det_wnd: Dict[str, list] = dict()

        for tracker in self.trackers:
            ref_wnd[tracker.name] = list()
            det_wnd[tracker.name] = list()

        return ref_wnd, det_wnd

    def __has_reached_limit(self, window: Dict[str, list], tracker: Tracker) -> bool:
        return len(window[tracker.name]) >= int(self.limit_per_window // 2)

    def  detect_drift(self, ref_wnd: Dict[str, list], det_wnd: Dict[str, int]):
        for tracker in self.trackers:
            ref_window_vals = np.array(ref_wnd[tracker.name])
            det_window_vals = np.array(det_wnd[tracker.name])

            for divergence_test in self.divergence_metrics:
                if divergence_test.is_divergent(ref_window_vals, det_window_vals):
                    self.ensemble_voting[f'{tracker.name}+{divergence_test.name}'] = 1
                else:
                    self.ensemble_voting[f'{tracker.name}+{divergence_test.name}'] = 0


    def __str__(self):
        return f'DynAmo:\tℓ={self.limit_per_window},\n\tτ={self.lookup_size},\n\tδ={self.wnd_moving_step},\n\tς={self.drift_detection_threshold}'

    def run(self) -> List[int]:
        signal: np.array = self.signal
        drift_window: np.array = np.array([0] * (signal.shape[0]))
        i: int = 0
        j: int = i + 1
        ref_wnd, det_wnd = self.__init_windows()
        
        start_time = time.time()

        prev_feature_boundaries = self.__calc_box(signal[max(0,i-self.lookup_size):j+1])

        while j < len(signal):
            x = signal[max(0, i - self.lookup_size):j+1]

            curr_feature_boundaries = self.__calc_box(x)

            for tracker in self.trackers:
                traced_feature_boundaries = tracker.track(prev_feature_boundaries, curr_feature_boundaries)
                limit_reached = self.__has_reached_limit(ref_wnd, tracker)

                if not limit_reached:
                    ref_wnd[tracker.name].append(traced_feature_boundaries)
                elif self.insert_reference_window > 1 - len(self.trackers):
                    self.insert_reference_window -= 1

            if self.insert_reference_window == 1 - len(self.trackers):
                for tracker in self.trackers:
                    traced_feature_boundaries = tracker.track(prev_feature_boundaries, curr_feature_boundaries)
                    limit_reached = self.__has_reached_limit(det_wnd, tracker)

                    if not limit_reached:
                        det_wnd[tracker.name].append(traced_feature_boundaries)
                    elif self.insert_detection_window > 1 - len(self.trackers):
                        self.insert_detection_window -= 1
            
            if self.insert_reference_window == 1 - len(self.trackers) and self.insert_detection_window == 1 - len(self.trackers):
                self.detect_drift(ref_wnd, det_wnd)

                if self.consensus_func.eval_drift(self.ensemble_voting, self.drift_detection_threshold):
                    drift_window[j-int(self.limit_per_window//2):j] = 1
                    i = j

                for tracker in self.trackers:
                    traced_feature_boundaries = tracker.track(prev_feature_boundaries, curr_feature_boundaries)
                    det_wnd[tracker.name].append(traced_feature_boundaries)

                    ref_wnd[tracker.name].append(det_wnd[tracker.name].pop(0))
                    ref_wnd[tracker.name].pop(0)
                
                self.insert_reference_window = 0
                self.insert_detection_window = 0

            prev_feature_boundaries = curr_feature_boundaries.copy()
            j += self.wnd_moving_step


        end_time = time.time()

        print(f'Elapsed time {end_time-start_time}')

        return drift_window