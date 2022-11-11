from typing import Dict, List
import numpy as np
from metrics import Metric, Recall, Precision, FScore



class DefaultEvalStrategy:

    def __init__(self,
        y_true: np.array,
        y_pred: np.array,
        eval_metrics: List[Metric] = [Recall(), Precision(), FScore()]):
        """ Instantiates a DefaultEvalStrategy object

        Parameters
        ----------
        y_true: np.array
            the ground truth labels

        y_pred: np.array
            the predicted outcome

        eval_metrics: List[Metric]
            a list containing the evaluation metrics to calculate
            on y_true and y_pred
            (default is [recall, precision, f1_score] with no margins).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.eval_metrics = eval_metrics

    def eval(self) -> Dict[str, float]:
        """ Calculates the evaluation metrics

        Return
        ----------
        Dict[str, float]:
            a dictionary of key-value pairs
            (name of evaluation metric, value of the evaluation metric).
        """
        res: Dict[str, float] = dict()
        for metric in self.eval_metrics:
            res[metric.name] = metric.calculate(self.y_true, self.y_pred)
        return res