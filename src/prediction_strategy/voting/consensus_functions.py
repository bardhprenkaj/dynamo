
import abc
from typing import Dict
import numpy as np
import pandas as pd

class Consensus(abc.ABC):

    @abc.abstractmethod
    def merge(self, votings: Dict[str, int]) -> int:
        pass

    def eval_drift(self, votings: Dict[str, int], threshold: float) -> bool:
        return self.merge(votings) >= threshold


class MajorityVoting(Consensus):
    
    def merge(self, votings: Dict[str, int]) -> int:
        votings_df = pd.Series(list(votings.values()))
        return votings_df.value_counts().index[0]


class AverageVoting(Consensus):

    def merge(self, votings: Dict[str, int]) -> int:
        values = list(votings.values())
        return sum(values) / len(values)
