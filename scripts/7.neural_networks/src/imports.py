# ---------------------------------------
# General Libs/Methods to manipulate data
# ---------------------------------------
import pandas as pd
import numpy as np
import os
#from src.custom.gtex import GTExMultiData, GtexPatient
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------
# Libs/Methods to work with deep learning
# ---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from src.custom import dataset
from src.custom import loss as ls

from src.custom import models
# ---------------------------------------
# Libs/Methods to track experiments and
# tuning hyperparameters
# ---------------------------------------
import optuna
import mlflow
import mlflow.pytorch
from optuna.integration.mlflow import MLflowCallback

# ---------------------------------------
# Libs/Methods for visualization
# ---------------------------------------
import matplotlib
import matplotlib.pylab as plt
from src.custom import plots

# ---------------------------------------
# Miscellaneous Libs/Methods
# ---------------------------------------
from tqdm import tqdm
import random
from src.custom import utils
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy