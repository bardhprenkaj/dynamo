import abc
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support, precision_recall_curve, auc

class Metric(abc.ABC):

    def __init__(self, margin: int=0):
        """ Instantiates a Metric object

        Parameters
        ----------
        margin: int
            the loose margin that copes with drift signalled
            in advance/delay (default is 0).
        """
        self.name = self.__class__.__name__
        self.margin = margin

    @abc.abstractmethod
    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        """ Calculates the evaluation metric.

        Parameters
        ----------
        y_true: np.array
            the ground truth labels

        y_pred : np.array
            the predicted outcomes

        Returns
        -------
        float:
            the metric calculated
        """
        pass

    def transform_predictions(self, y_pred: np.array) -> np.array:
        """ Addresses the loose margins for the prediction.
        In this way, one can have a safe boundary of predictions where
        a drift is signalled margin-steps in advance/delay

        Parameters
        ----------
        y_pred : np.array
            the predicted outcomes

        Returns
        -------
        np.array
            the modified y_pred array with the loose margin
        """
        if self.margin == 0:
            return y_pred

        where_are_the_anomalies = np.where(y_pred == 1)[0]

        res = np.zeros(y_pred.shape[0])

        for index in where_are_the_anomalies:
            res[index:index+self.margin] = 1
            if index - self.margin >= 0:
                res[index-self.margin:index] = 1

        return res


class Precision(Metric):

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        y_pred = super().transform_predictions(y_pred)
        return precision_score(y_true, y_pred)
    
class Recall(Metric):

    def __init__(self, margin: int = 0, pos_label: int = 1):
        super().__init__(margin=margin)
        self.pos_label = pos_label

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        y_pred = super().transform_predictions(y_pred)
        return recall_score(y_true, y_pred, pos_label=self.pos_label)

class FBetaScore(Metric):

    def __init__(self, margin: int = 0, beta: float = 1):
        """ Instantiates a FBetaScore object

        Parameters
        ----------
        margin: int
            the loose margin that copes with drift signalled
            in advance/delay (default is 0).

        beta: float
            a positive real factor  chosen such that recall is considered beta  times as important as precision
            F_beta = (1 + beta^2) * (precision * recall)/((beta^2 * precision) + recall)
        """
        super().__init__(margin=margin)

        self.beta = beta
        self.recall = Recall(margin=margin)
        self.precision = Precision(margin=margin)

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        rec = self.recall.calculate(y_true, y_pred)
        prec = self.precision.calculate(y_true, y_pred)

        if prec == 0 and rec == 0:
            return 0

        return (1 + self.beta**2) * (rec * prec) / ((self.beta**2 * prec) + rec)


class F1Score(Metric):

    def __init__(self, margin: int = 0):
        super().__init__(margin=margin)

        self.margin = margin

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        return FBetaScore(margin=self.margin).calculate(y_true, y_pred)

class MacroF1Score(Metric):

    def __init__(self, margin: int = 0):
        """
            Instantiates a MacroF1Score object

            Parameters
            ----------
            margin: int
                the loose margin that copes with drift signalled
                in advance/delay (default is 0).        
        """
        super().__init__(margin=margin)

    
    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred)
        return np.mean(f)

class AUCPR(Metric):

    def __init__(self, margin: int = 0):
        super().__init__(margin=margin)

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        y_pred = super().transform_predictions(y_pred)
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
        return auc(recalls, precisions)


class InverseWeightedF1Score(Metric):

    def __init__(self, margin: int = 0):
        super().__init__(margin=margin)


    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        y_pred = super().transform_predictions(y_pred)
        _, _, f, sup = precision_recall_fscore_support(y_true, y_pred)
        sup = sup / np.sum(sup)
        return f[0] * (1-sup[0]) + f[1] * (1-sup[1])
    

class FalsePositiveRate(Metric):

    def __init__(self, margin: int = 0):
        super().__init__(margin=margin)
        self.recall = Recall(pos_label=0)

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        return 1 - self.recall.calculate(y_true, y_pred)
    
class FalseNegativeRate(Metric):

    def __init__(self, margin: int = 0):
        super().__init__(margin=margin)
        self.recall = Recall()

    def calculate(self, y_true: np.array, y_pred: np.array) -> float:
        return 1 - self.recall.calculate(y_true, y_pred)