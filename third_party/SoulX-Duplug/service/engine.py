import math
import numpy as np
import torch


class TurnTakingEngine:
    def __init__(
        self,
        model,
    ):
        """
        model: TurnModel (model.py)
        """
        self.model = model
        self.device = model.device
        self.contexts = None

    def process(self, audio: np.ndarray):
        """
        audio: np.float32, shape [N]
        return:
            None | state_token(str)
        """
        if self.contexts is None:
            self.model.reset()
        else:
            self.model.restore_runtime(self.contexts)

        result = self.model.process(audio)
        self.contexts = self.model.snapshot_runtime()

        return result
