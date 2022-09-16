"""
Three main functionality in network module
net_train: training a network
net_test: testing a network to find metrics of the concatenated trajectories
"""
from .model_factory import get_model
from .test import net_test
from .train import net_train
