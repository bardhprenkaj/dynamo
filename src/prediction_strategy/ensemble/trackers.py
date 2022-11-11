import numpy as np


class Tracker:
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.tracked_events = 0

    def track(self, prev_box: np.array, curr_box: np.array) -> np.array:
        self.tracked_events += 1

    def reset_trace(self):
        self.tracked_events = 0


class BoxSizeTracker(Tracker):

    def track(self, prev_box: np.array, curr_box: np.array) -> np.array:
        super().track(prev_box, curr_box)
        return np.diff(curr_box, axis=1).squeeze()


class BoxSizeProductTracker(BoxSizeTracker):

    def track(self, prev_box: np.array, curr_box: np.array) -> np.array:
        return np.prod(super().track(prev_box, curr_box))

class NormalizedBoxSizeTracker(BoxSizeTracker):

    def track(self, prev_box: np.array, curr_box: np.array) -> np.array:
        return np.linalg.norm(super().track(prev_box, curr_box))


class DifferenceBoxTracker(BoxSizeTracker):

    def track(self, prev_box: np.array, curr_box: np.array) -> np.array:
        return super().track(prev_box, curr_box-prev_box)

class NormalizedDifferenceBoxTracker(DifferenceBoxTracker):

    def track(self, prev_box: np.array, curr_box: np.array) -> np.array:
        return np.linalg.norm(super().track(prev_box, curr_box))