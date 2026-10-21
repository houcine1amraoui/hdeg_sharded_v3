import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Converts a multivariate time series into
    window-to-window forecasting samples.

    Each sample consists of:

        Input :
            X_t = data[t : t + W]

        Target :
            X_{t+1} = data[t + 1 : t + W + 1]

    where

        W = sliding window length
        N = number of devices

    Therefore,

        Input shape  : [W, N]
        Target shape : [W, N]

    This formulation enables HDEG to learn the
    temporal evolution of smart-home behaviour
    over an entire future observation window,
    rather than forecasting only a single
    future timestamp.
    """

    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.T = data.shape[0]

    def __len__(self):
        # Need one complete future window
        return self.T - self.window_size

    def __getitem__(self, idx):

        # Current observation window
        x = self.data[idx : idx + self.window_size]

        # Future observation window (shifted by one timestep)
        y = self.data[idx + 1 : idx + self.window_size + 1]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y