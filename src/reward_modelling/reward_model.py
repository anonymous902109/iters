import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reward_modelling.replay_buffer import ReplayBuffer
from src.reward_modelling.reward_nn import RewardModelNN


class RewardModel:

    def __init__(self, time_window, input_size):
        self.time_window = time_window

        self.buffer = ReplayBuffer(capacity=10000, time_window=self.time_window)
        self.predictor = RewardModelNN(input_size)

    def update(self):
        dataset = self.buffer.get_dataset()
        train, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

        self.predictor.train(DataLoader(train, shuffle=True, batch_size=512))
        self.predictor.evaluate(DataLoader(test, shuffle=True, batch_size=512))

    def update_buffer(self, D, signal, important_features, datatype, actions, rules, iter):
        self.buffer.update(D, signal, important_features, datatype, actions, rules, iter)

    def predict(self, encoding):
        encoding = np.array(encoding).reshape(1, -1)
        return self.predictor.predict(encoding)

    def save(self):
        self.predictor.save()




