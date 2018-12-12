import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader.data_loader import DataLoader


class DataGenerator:
    def __init__(self, config, filepath):
        self.config = config
        self.data = DataLoader(filepath)
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
