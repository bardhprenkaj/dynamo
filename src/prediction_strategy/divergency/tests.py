
import abc
import numpy as np

class DivergenceMetric(abc.ABC):

    def __init__(self):
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def is_divergent(self, reference_wnd: np.array, detection_wnd: np.array) -> bool:
        pass

class DensestHyperboxDifference(DivergenceMetric):

    def is_divergent(self, reference_wnd: np.array, detection_wnd: np.array) -> bool:
        return np.sum(np.abs(np.diff(detection_wnd))) > np.sum(np.abs(np.diff(reference_wnd)))


class MeanDivergence(DivergenceMetric):

    def is_divergent(self, reference_wnd: np.array, detection_wnd: np.array) -> bool:
        detection_wnd_mean, detection_wnd_std = detection_wnd.mean(axis=0), detection_wnd.std(axis=0)
        reference_wnd_mean = reference_wnd.mean(axis=0)

        bounds = (detection_wnd_mean - detection_wnd_std, detection_wnd_mean + detection_wnd_std) 
        test = (reference_wnd_mean < bounds[0]).any() or (reference_wnd_mean > bounds[1]).any()
        return test